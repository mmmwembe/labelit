import os
import tempfile
import requests
import hashlib
import shutil
import json
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlparse
import fitz  # PyMuPDF
from google.cloud import storage
from dotenv import load_dotenv


class PDFOps:
    """
    A class for handling PDF operations including text extraction, image extraction,
    and Google Cloud Storage interactions.
    """
    
    def __init__(self):
        """Initialize PDFOps with Google Cloud credentials"""
        load_dotenv()
        self.secret_json = os.getenv('GOOGLE_SECRET_JSON')
        if not self.secret_json:
            raise ValueError("GOOGLE_SECRET_JSON environment variable not found")
    
    def _get_storage_client(self) -> storage.Client:
        """
        Get authenticated Google Cloud Storage client.
        
        Returns:
            storage.Client: Authenticated GCS client
        """
        return storage.Client.from_service_account_info(json.loads(self.secret_json))
    
    @staticmethod
    def _get_file_hash(file_content: bytes) -> str:
        """
        Calculate SHA-256 hash of file content.
        
        Args:
            file_content (bytes): Content to hash
            
        Returns:
            str: SHA-256 hash of the content
        """
        return hashlib.sha256(file_content).hexdigest()
    
    @staticmethod
    def _download_pdf(pdf_url: str, temp_path: str) -> bytes:
        """
        Download PDF from URL and save to temporary path.
        
        Args:
            pdf_url (str): URL of the PDF file
            temp_path (str): Path to save the temporary file
            
        Returns:
            bytes: Raw PDF content
            
        Raises:
            requests.exceptions.RequestException: If download fails
        """
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        with open(temp_path, 'wb') as f:
            f.write(response.content)
            
        return response.content
    
    def extract_text_from_pdf(self, pdf_url: str) -> Tuple[str, str, str]:
        """
        Downloads PDF from URL to temp directory, extracts text content, and returns
        the full text, first two pages of text, and filename.
        
        Args:
            pdf_url (str): The URL of the PDF file
        
        Returns:
            Tuple[str, str, str]: A tuple containing:
                - entire_pdf_text_content: Complete text content from all pages
                - first_two_pages_text_content: Text content from first two pages only
                - filename: Original filename from the PDF URL
        """
        temp_dir = None
        try:
            # Extract filename from URL
            parsed_url = urlparse(pdf_url)
            filename = os.path.basename(parsed_url.path)
            if not filename.lower().endswith('.pdf'):
                filename = 'unnamed.pdf'
            
            # Create temporary directory and file
            temp_dir = tempfile.mkdtemp()
            temp_pdf_path = os.path.join(temp_dir, 'temp.pdf')
            
            # Download PDF
            self._download_pdf(pdf_url, temp_pdf_path)
            
            # Extract text using PyMuPDF
            doc = fitz.open(temp_pdf_path)
            
            try:
                # Process all pages for complete text
                entire_pdf_text_content = ""
                for page in doc:
                    entire_pdf_text_content += page.get_text()
                
                # Process first two pages separately
                first_two_pages_text_content = ""
                for page_num in range(min(2, doc.page_count)):
                    first_two_pages_text_content += doc[page_num].get_text()
                
                return entire_pdf_text_content, first_two_pages_text_content, filename
                
            finally:
                doc.close()
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return "", "", ""
            
        finally:
            # Clean up temp directory and its contents
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def upload_to_gcs(self, image_content: bytes, filename: str, session_id: str, bucket_name: str) -> Optional[str]:
        """
        Upload image to Google Cloud Storage and return public URL.
        
        Args:
            image_content (bytes): The image content to upload
            filename (str): Name for the file in storage
            session_id (str): Session identifier for organizing uploads
            bucket_name (str): Name of the GCS bucket
            
        Returns:
            Optional[str]: Public URL of uploaded image or None if upload fails
        """
        try:
            client = self._get_storage_client()
            bucket = client.bucket(bucket_name)

            # Create blob path using session ID and filename
            blob_path = f"{session_id}/{filename}"
            blob = bucket.blob(blob_path)

            # Upload image
            blob.upload_from_string(image_content, content_type='image/jpeg')

            # Generate public URL
            return f"https://storage.googleapis.com/{bucket_name}/{blob_path}"

        except Exception as e:
            print(f"Error uploading to GCS: {str(e)}")
            return None

    def extract_images_and_metadata(self, pdf_url: str, session_id: str, bucket_name: str) -> Optional[Dict[str, Any]]:
        """
        Extracts images and metadata from a PDF file.
        
        Args:
            pdf_url (str): URL of the PDF file
            session_id (str): Session identifier for organizing uploads
            bucket_name (str): Name of the GCS bucket for storing images
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary containing extracted metadata and image URLs
        """
        temp_dir = None
        temp_path = None
        
        try:
            # Convert GCS URL to direct download URL if needed
            pdf_url = pdf_url.replace("storage.cloud.google.com", "storage.googleapis.com")

            # Create temporary directory and file
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, 'temp.pdf')
            
            # Download PDF and get content
            pdf_content = self._download_pdf(pdf_url, temp_path)
            file_256_hash = self._get_file_hash(pdf_content)

            # Open PDF with PyMuPDF
            pdf_document = fitz.open(temp_path)
            
            try:
                total_pages = len(pdf_document)
                
                # Initialize result structure
                result = {
                    "file_256_hash": file_256_hash,
                    "images_in_doc": [],
                    "paper_image_urls": [],
                    "total_images": 0,
                    "page_details": []
                }

                # Process each page
                for page_num in range(total_pages):
                    page = pdf_document[page_num]
                    image_list = page.get_images()

                    page_info = {
                        "page_index": page_num,
                        "total_pages": total_pages,
                        "has_images": len(image_list) > 0,
                        "num_images": len(image_list),
                        "image_urls": []
                    }

                    if image_list:
                        for img_idx, img in enumerate(image_list, 1):
                            try:
                                # Extract and upload image
                                xref = img[0]
                                base_image = pdf_document.extract_image(xref)
                                image_filename = f"{file_256_hash}_image_{img_idx}.jpeg"
                                
                                image_url = self.upload_to_gcs(
                                    image_content=base_image["image"],
                                    filename=image_filename,
                                    session_id=session_id,
                                    bucket_name=bucket_name
                                )

                                if image_url:
                                    page_info["image_urls"].append(image_url)
                                    result["paper_image_urls"].append(image_url)

                            except Exception as e:
                                print(f"Error processing image {img_idx} on page {page_num + 1}: {str(e)}")

                    # Update total_images count and append page info
                    result["total_images"] += page_info["num_images"]
                    result["images_in_doc"].append(page_info)

                    if page_info["has_images"]:
                        result["page_details"].append({
                            "page_index": page_num,
                            "num_images": page_info["num_images"],
                            "image_urls": page_info["image_urls"]
                        })

                return result

            finally:
                pdf_document.close()

        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return None
            
        finally:
            # Clean up temporary files
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


# Example usage:
# pdf_ops = PDFOps()
#
# # Extract text and get first two pages preview
# full_text, first_two_pages, filename = pdf_ops.extract_text_from_pdf("https://example.com/sample.pdf")
# print(f"Filename: {filename}")
# print(f"First two pages preview:\n{first_two_pages[:500]}...")
#
# # Extract images and metadata
# result = pdf_ops.extract_images_and_metadata(
#     pdf_url="https://example.com/sample.pdf",
#     session_id="unique_session_id",
#     bucket_name="your-gcs-bucket-name"
# )