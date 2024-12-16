import os
import json
import logging
from google.cloud import storage
from dotenv import load_dotenv
import pandas as pd
import requests
from typing import List, Dict, Any, Optional, Union
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GCPOps:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Get the Google service account JSON from environment variable
        self.secret_json = os.getenv('GOOGLE_SECRET_JSON')
        try:
            self.storage_client = storage.Client.from_service_account_info(
                json.loads(self.secret_json)
            )
        except Exception as e:
            logger.error(f"Failed to initialize GCP storage client: {str(e)}")
            raise

    def save_file_to_bucket(self, artifact_url: str, session_id: str, bucket_name: str, 
                          subdir: str = "papers", 
                          subsubdirs: List[str] = ["pdf","word","images","csv","text"]) -> Optional[str]:
        """
        Save a file to a GCP bucket with appropriate subdirectory structure.
        """
        try:
            # Determine the subsubdir based on the file extension
            if artifact_url.endswith(".docx"):
                subsubdir = "word"
            elif artifact_url.endswith((".jpg", ".jpeg", ".png", ".gif")):
                subsubdir = "images"
            elif artifact_url.endswith(".csv"):
                subsubdir = "csv"
            elif artifact_url.endswith((".txt", ".text")):
                subsubdir = "text"
            else:
                subsubdir = "pdf"  # Default to PDF if no other match

            bucket = self.storage_client.bucket(bucket_name)

            if subsubdir == "word":
                # Delete the contents of the subsubdir before uploading the new file
                blob_prefix = f"{session_id}/{subdir}/{subsubdir}/"
                blobs = self.storage_client.list_blobs(bucket, prefix=blob_prefix)
                for blob in blobs:
                    blob.delete()

            # Upload the file to Google Cloud Storage
            blob_name = f"{session_id}/{subdir}/{subsubdir}/{os.path.basename(artifact_url)}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(artifact_url)
            return blob.public_url
            
        except Exception as e:
            logger.error(f"Error saving file to bucket: {str(e)}")
            return None

    def save_tracker_csv(self, df: pd.DataFrame, session_id: str, bucket_name: str) -> Optional[str]:
        """
        Save a pandas DataFrame as a CSV to the specified GCS bucket.
        """
        try:
            # Create a temporary local CSV file
            temp_csv_path = os.path.join('static', 'tmp', f'{session_id}.csv')
            os.makedirs(os.path.dirname(temp_csv_path), exist_ok=True)
            
            # Save DataFrame to temporary CSV
            df.to_csv(temp_csv_path, index=False)
            
            try:
                # Use save_file_to_bucket to upload the CSV
                url = self.save_file_to_bucket(
                    artifact_url=temp_csv_path,
                    session_id=session_id,
                    bucket_name=bucket_name
                )
                return url
            finally:
                # Clean up temporary file
                if os.path.exists(temp_csv_path):
                    os.remove(temp_csv_path)
            
        except Exception as e:
            logger.error(f"Error saving tracker CSV: {str(e)}")
            return None

    def initialize_paper_upload_tracker_df_from_gcp(self, session_id: str, bucket_name: str) -> pd.DataFrame:
        """
        Initialize a pandas DataFrame from a CSV file stored in GCS.
        """
        try:
            # Construct the GCS URL
            gcs_url = f"https://storage.googleapis.com/{bucket_name}/{session_id}/papers/csv/{session_id}.csv"
            
            # Read directly from URL
            df = pd.read_csv(gcs_url)
            logger.info(f"Successfully loaded tracker DataFrame with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error initializing DataFrame from GCS: {str(e)}")
            # If file doesn't exist or other error, return empty DataFrame with default columns
            return pd.DataFrame(columns=[
                'gcp_public_url',
                'hash',
                'original_filename',
                'citation_name',
                'citation_authors',
                'citation_year',
                'citation_organization',
                'citation_doi',
                'citation_url',
                'upload_timestamp',
                'processed',
            ])

    def get_public_urls(self, bucket_name: str, session_id: str, file_hash_num: str) -> List[str]:
        """
        Get public URLs for all files in a specific bucket path.
        """
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=f"{session_id}/{file_hash_num}/")
            return [f"https://storage.googleapis.com/{bucket_name}/{blob.name}" for blob in blobs]
        except Exception as e:
            logger.error(f"Error getting public URLs: {str(e)}")
            return []

    def get_public_urls_with_metadata(self, bucket_name: str, session_id: str, 
                                    file_hash_num: str) -> List[Dict[str, Any]]:
        """
        Get public URLs and metadata for all files in a specific bucket path.
        """
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=f"{session_id}/{file_hash_num}/")
            
            files = []
            for blob in blobs:
                file_info = {
                    'name': blob.name.split('/')[-1],  # File name
                    'blob_name': blob.name,  # Full blob name
                    'size': f"{blob.size / 1024 / 1024:.2f} MB",  # Size in MB
                    'updated': blob.updated.strftime('%Y-%m-%d %H:%M:%S'),  # Last updated timestamp
                    'public_url': f"https://storage.googleapis.com/{bucket_name}/{blob.name}"  # Public URL
                }
                files.append(file_info)
            
            return files
            
        except Exception as e:
            logger.error(f"Error getting public URLs with metadata: {str(e)}")
            return []

    def load_paper_json_files(self, papers_json_public_url: str) -> List[Dict[str, Any]]:
        """
        Load existing paper JSON files from GCS with enhanced error handling and validation.
        """
        try:
            bucket_name = papers_json_public_url.split('/')[3]
            blob_path = '/'.join(papers_json_public_url.split('/')[4:])
            
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            if blob.exists():
                content = blob.download_as_string()
                data = json.loads(content)
                
                # Validate and process each paper's data
                processed_data = []
                for paper in data:
                    if isinstance(paper.get('diatoms_data'), str):
                        try:
                            paper['diatoms_data'] = json.loads(paper['diatoms_data'])
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping paper with invalid diatoms_data JSON")
                            continue
                    processed_data.append(paper)
                
                logger.info(f"Successfully loaded {len(processed_data)} papers from GCS")
                return processed_data
            else:
                logger.warning(f"No file found at {papers_json_public_url}")
                return []
                
        except Exception as e:
            logger.error(f"Error loading paper JSON files: {str(e)}")
            return []

    def save_paper_json_files(self, papers_json_public_url: str, 
                            paper_json_files: List[Dict[str, Any]]) -> str:
        """
        Save paper JSON files to GCS with enhanced error handling and validation.
        """
        try:
            # Validate and process input data
            processed_files = []
            for paper in paper_json_files:
                # Ensure diatoms_data is properly formatted
                if isinstance(paper.get('diatoms_data'), str):
                    try:
                        paper['diatoms_data'] = json.loads(paper['diatoms_data'])
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping paper with invalid diatoms_data JSON")
                        continue
                processed_files.append(paper)
            
            bucket_name = papers_json_public_url.split('/')[3]
            blob_path = '/'.join(papers_json_public_url.split('/')[4:])
            
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            # Save with proper formatting and content type
            json_content = json.dumps(processed_files, indent=2)
            blob.upload_from_string(
                json_content,
                content_type='application/json'
            )
            
            logger.info(f"Successfully saved {len(processed_files)} papers to GCS")
            return papers_json_public_url
            
        except Exception as e:
            logger.error(f"Error saving paper JSON files: {str(e)}")
            return ""

    def save_json_to_bucket(self, local_file_path: str, bucket_name: str, 
                          session_id: str) -> Optional[str]:
        """
        Save a local JSON file to a GCP bucket with validation.
        """
        try:
            # Validate input JSON
            with open(local_file_path, 'r') as f:
                data = json.load(f)  # This will raise JSONDecodeError if invalid
            
            bucket = self.storage_client.bucket(bucket_name)
            blob_name = f"labels/{session_id}/{session_id}.json"
            blob = bucket.blob(blob_name)

            blob.upload_from_filename(local_file_path)
            public_url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
            
            logger.info(f"Successfully saved JSON to {public_url}")
            return public_url
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {local_file_path}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error saving JSON to bucket: {str(e)}")
            return None

    @staticmethod
    def check_gcs_file_exists(url: str) -> bool:
        """
        Check if a file exists in Google Cloud Storage using the public URL.
        """
        try:
            response = requests.head(url, timeout=10)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Error checking file existence: {str(e)}")
            return False

    def validate_and_process_paper_json(self, paper_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and process a paper JSON object to ensure proper format.
        """
        try:
            # Ensure diatoms_data is a dictionary
            if isinstance(paper_json.get('diatoms_data'), str):
                paper_json['diatoms_data'] = json.loads(paper_json['diatoms_data'])
            
            # Ensure required fields exist
            required_fields = ['image_url', 'info']
            if not all(field in paper_json['diatoms_data'] for field in required_fields):
                raise ValueError("Missing required fields in diatoms_data")
            
            # Validate info array
            for info in paper_json['diatoms_data']['info']:
                if not isinstance(info.get('label'), list):
                    info['label'] = [str(info.get('label', ''))]
                
                # Ensure bbox and yolo_bbox are strings
                info['bbox'] = str(info.get('bbox', ''))
                info['yolo_bbox'] = str(info.get('yolo_bbox', ''))
            
            return paper_json
            
        except Exception as e:
            logger.error(f"Error validating paper JSON: {str(e)}")
            raise

    def sync_paper_json_files(self, papers_json_public_url: str, 
                            updated_data: Dict[str, Any], 
                            image_index: int) -> bool:
        """
        Synchronize updated paper JSON files with GCS storage.
        """
        try:
            # Load existing files
            paper_json_files = self.load_paper_json_files(papers_json_public_url)
            
            # Update the specific paper's data
            for paper in paper_json_files:
                if isinstance(paper.get('diatoms_data'), str):
                    paper['diatoms_data'] = json.loads(paper['diatoms_data'])
                
                current_data = paper.get('diatoms_data', {})
                if current_data.get('image_url') == updated_data.get('image_url'):
                    paper['diatoms_data'] = self.validate_and_process_paper_json({
                        'diatoms_data': updated_data
                    })['diatoms_data']
                    break
            
            # Save updated files
            result_url = self.save_paper_json_files(papers_json_public_url, paper_json_files)
            return bool(result_url)
            
        except Exception as e:
            logger.error(f"Error syncing paper JSON files: {str(e)}")
            return False  

    # Add these methods to the GCPOps class in gcpOps.py

    # def save_segmentation_data(self, segmentation_data: str, image_filename: str, 
    #                         session_id: str, bucket_name: str) -> Optional[str]:
    #     """
    #     Save segmentation data to a text file in GCS bucket.
        
    #     Args:
    #         segmentation_data: String containing the segmentation data
    #         image_filename: Base filename of the image being segmented
    #         session_id: Current session ID
    #         bucket_name: Name of the GCS bucket for segmentation data
            
    #     Returns:
    #         Optional[str]: Public URL of the saved segmentation file, or None if error
    #     """
    #     try:
    #         # Create temporary file
    #         with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
    #             temp_file.write(segmentation_data)
    #             temp_path = temp_file.name
                
    #         try:
    #             # Upload to GCS
    #             bucket = self.storage_client.bucket(bucket_name)
    #             blob_name = f"{session_id}/{image_filename}.txt"
    #             blob = bucket.blob(blob_name)
                
    #             blob.upload_from_filename(temp_path)
                
    #             # Generate public URL
    #             public_url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
    #             logger.info(f"Saved segmentation data to {public_url}")
                
    #             return public_url
                
    #         finally:
    #             # Clean up temp file
    #             if os.path.exists(temp_path):
    #                 os.remove(temp_path)
                    
    #     except Exception as e:
    #         logger.error(f"Error saving segmentation data: {str(e)}")
    #         return None
            
    # def load_segmentation_data(self, segmentation_url: str) -> Optional[str]:
    #     """
    #     Load segmentation data from GCS.
        
    #     Args:
    #         segmentation_url: Public URL of the segmentation file
            
    #     Returns:
    #         Optional[str]: Content of the segmentation file, or None if error
    #     """
    #     try:
    #         bucket_name = segmentation_url.split('/')[3]
    #         blob_path = '/'.join(segmentation_url.split('/')[4:])
            
    #         bucket = self.storage_client.bucket(bucket_name)
    #         blob = bucket.blob(blob_path)
            
    #         if blob.exists():
    #             return blob.download_as_string().decode('utf-8')
    #         else:
    #             logger.warning(f"No segmentation file found at {segmentation_url}")
    #             return None
                
    #     except Exception as e:
    #         logger.error(f"Error loading segmentation data: {str(e)}")
    #         return None



    def save_segmentation_data(self, segmentation_data: str, image_filename: str, 
                            session_id: str, bucket_name: str) -> Optional[str]:
        """
        Save segmentation data to a text file in GCS bucket.
        
        Args:
            segmentation_data: String containing the segmentation data
            image_filename: Base filename of the image being segmented
            session_id: Current session ID
            bucket_name: Name of the GCS bucket for segmentation data
            
        Returns:
            Optional[str]: Public URL of the saved segmentation file, or None if error
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                temp_file.write(segmentation_data)
                temp_path = temp_file.name
                
            try:
                # Upload to GCS
                bucket = self.storage_client.bucket(bucket_name)
                blob_name = f"{session_id}/{image_filename}.txt"
                blob = bucket.blob(blob_name)
                
                blob.upload_from_filename(temp_path)
                
                # Generate public URL
                public_url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
                logger.info(f"Saved segmentation data to {public_url}")
                
                return public_url
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            logger.error(f"Error saving segmentation data: {str(e)}")
            return None
            
    def load_segmentation_data(self, segmentation_url: str) -> Optional[str]:
        """
        Load segmentation data from GCS.
        
        Args:
            segmentation_url: Public URL of the segmentation file
            
        Returns:
            Optional[str]: Content of the segmentation file, or None if error
        """
        try:
            bucket_name = segmentation_url.split('/')[3]
            blob_path = '/'.join(segmentation_url.split('/')[4:])
            
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            if blob.exists():
                return blob.download_as_string().decode('utf-8')
            else:
                logger.warning(f"No segmentation file found at {segmentation_url}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading segmentation data: {str(e)}")
            return None
        
    def get_segmentation_data(self, filename, bucket_name):
        """Get segmentation data from GCS bucket"""
        try:
            bucket = self.storage_client.get_bucket(bucket_name)
            blob = bucket.blob(filename)
            
            if not blob.exists():
                return None
                
            return blob.download_as_string().decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error getting segmentation data: {str(e)}")
            return None

    def get_uploaded_files(self, bucket_name, session_id):
        try:
            bucket = self.storage_client.get_bucket(bucket_name)
            prefix = f"pdf/{session_id}/"
            blobs = bucket.list_blobs(prefix=prefix)
            
            files = []
            for blob in blobs:
                file_info = {
                    'name': blob.name.split('/')[-1],
                    'blob_name': blob.name, 
                    'size': f"{blob.size / 1024 / 1024:.2f} MB",
                    'updated': blob.updated.strftime('%Y-%m-%d %H:%M:%S'),
                    'public_url': f"https://storage.googleapis.com/{bucket_name}/{blob.name}"
                }
                files.append(file_info)
            
            return sorted(files, key=lambda x: x['updated'], reverse=True)
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
        
    def get_blob_content(self, bucket_name, blob_name):
        try:
            bucket = self.storage_client.get_bucket(bucket_name)
            blob = bucket.blob(blob_name)
            return blob.download_as_bytes()
        except Exception as e:
            print(f"Error downloading blob: {e}")
            return None


    def save_pdf_file_to_bucket(self, local_file_path, bucket_name, session_id):
        try:
            filename = os.path.basename(local_file_path)
            bucket = self.storage_client.get_bucket(bucket_name)
            blob_name = f"pdf/{session_id}/{filename}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_file_path)
            public_url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
            return blob_name, public_url
        except Exception as e:
            print(f"Error uploading file: {e}") 
            return None
        

    def update_paper_json_files(self, PAPER_JSON_FILES, TEMP_JSON_FILES):
        """
        Update PAPER_JSON_FILES by adding elements from TEMP_JSON_FILES.
        If an element with the same 'pdf_file_url' exists in PAPER_JSON_FILES, it will be overwritten.

        :param PAPER_JSON_FILES: List of existing paper JSON objects
        :param TEMP_JSON_FILES: List of temporary paper JSON objects to add or update
        :return: Updated list of PAPER_JSON_FILES
        """
        # Create a dictionary for faster lookups by pdf_file_url in PAPER_JSON_FILES
        paper_files_dict = {paper['pdf_file_url']: paper for paper in PAPER_JSON_FILES}

        # Iterate through TEMP_JSON_FILES and update or append to PAPER_JSON_FILES
        for temp_paper in TEMP_JSON_FILES:
            pdf_url = temp_paper['pdf_file_url']
            # Update or add the paper to the dictionary
            paper_files_dict[pdf_url] = temp_paper

        # Convert the dictionary back to a list
        updated_paper_json_files = list(paper_files_dict.values())

        return updated_paper_json_files
    
    
    
    # Example usage
    # PAPER_JSON_FILES = gcp_ops.load_paper_json_files(papers_json_public_url)
    # TEMP_JSON_FILES = [
    #     {"pdf_file_url": "https://example.com/image2.pdf", "new_field": "new_value"},
    #     {"pdf_file_url": "https://example.com/image5.pdf", "new_field": "new_value"}
    # ]

    # # Call the function to update PAPER_JSON_FILES
    # PAPER_JSON_FILES = update_paper_json_files(PAPER_JSON_FILES, TEMP_JSON_FILES)

    # # Save the updated list back to the GCP bucket
    # gcp_ops.save_paper_json_files(papers_json_public_url, PAPER_JSON_FILES)