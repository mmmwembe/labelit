from flask import Flask, render_template, send_file, request, jsonify, send_from_directory
import json
import logging
import os
import tempfile
from datetime import datetime
from modules import ClaudeAI, GCPOps, PDFOps, SegmentationOps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Constants
SESSION_ID = 'eb9db0ca54e94dbc82cffdab497cde13'
PAPERS_BUCKET = 'papers-diatoms'
PAPERS_PROCESSED_BUCKET = 'papers-diatoms-processed'
PAPERS_BUCKET_LABELLING = 'papers-diatoms-labelling'
PAPERS_BUCKET_JSON_FILES = 'papers-diatoms-jsons'
BUCKET_EXTRACTED_IMAGES = 'papers-extracted-images-bucket-mmm'
BUCKET_PAPER_TRACKER_CSV = 'papers-extracted-pages-csv-bucket-mmm'
BUCKET_SEGMENTATION_LABELS = 'papers-diatoms-segmentation'

# Global variables for data management
PAPERS_JSON_PUBLIC_URL = f"https://storage.googleapis.com/{PAPERS_BUCKET_JSON_FILES}/jsons_from_pdfs/{SESSION_ID}/{SESSION_ID}.json"
PAPER_JSON_FILES = []
DIATOMS_DATA = []

# Initialize GCP operations
gcp_ops = GCPOps()

def initialize_data():
    """Initialize paper data and diatoms data"""
    global PAPER_JSON_FILES, DIATOMS_DATA
    
    try:
        PAPER_JSON_FILES = gcp_ops.load_paper_json_files(PAPERS_JSON_PUBLIC_URL)
        if PAPER_JSON_FILES:
            DIATOMS_DATA = ClaudeAI.get_DIATOMS_DATA(PAPERS_JSON_PUBLIC_URL)
            logger.info(f"Successfully loaded {len(DIATOMS_DATA)} diatom entries")
        else:
            logger.warning("No paper JSON files found")
            PAPER_JSON_FILES = []
            DIATOMS_DATA = []
    except Exception as e:
        logger.error(f"Error loading paper data: {str(e)}")
        PAPER_JSON_FILES = []
        DIATOMS_DATA = []

def save_labels(updated_data):
    """Save updated labels and synchronize data structures"""
    global PAPER_JSON_FILES, DIATOMS_DATA
    
    try:
        image_index = updated_data.get('image_index', 0)
        info = updated_data.get('info', [])
        
        if 0 <= image_index < len(DIATOMS_DATA):
            DIATOMS_DATA[image_index]['info'] = info
            
            for paper in PAPER_JSON_FILES:
                if 'diatoms_data' in paper:
                    if isinstance(paper['diatoms_data'], str):
                        paper['diatoms_data'] = json.loads(paper['diatoms_data'])
                    
                    if paper['diatoms_data'].get('image_url') == DIATOMS_DATA[image_index].get('image_url'):
                        paper['diatoms_data']['info'] = info
                        break
            
            success = ClaudeAI.update_and_save_papers(
                PAPERS_JSON_PUBLIC_URL,
                PAPER_JSON_FILES,
                DIATOMS_DATA
            )
            
            if not success:
                raise Exception("Failed to save updates to GCS")
                
            return True
            
    except Exception as e:
        logger.error(f"Error saving labels: {str(e)}")
        raise
        
    return False

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/modules')
def modules():
    """Route to display installed modules"""
    dummy_packages = {
        'Flask': '2.0.1',
        'Pandas': '1.3.3',
        'NumPy': '1.21.2',
        'Claude AI': '1.0.0',
        'GCP Tools': '2.1.0'
    }
    return render_template('modules.html', packages=dummy_packages)

@app.route('/all_papers')
def all_papers():
    """Route to display all papers"""
    dummy_pdf_urls = [
        'https://storage.googleapis.com/papers-diatoms/paper1.pdf',
        'https://storage.googleapis.com/papers-diatoms/paper2.pdf',
        'https://storage.googleapis.com/papers-diatoms/paper3.pdf'
    ]
    return render_template('papers.html', pdf_urls=dummy_pdf_urls)

@app.route('/view_uploaded_pdfs')
def view_uploaded_pdfs():
    """Route to view uploaded PDFs"""
    try:
        return send_from_directory('templates', 'view_uploaded_pdfs.html')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/diatoms_data')
def see_diatoms_data():
    """Route to display diatoms data"""
    try:
        if not PAPERS_BUCKET_JSON_FILES or not SESSION_ID:
            raise ValueError("Required configuration variables are not set")
        
        papers_json_public_url = f"https://storage.googleapis.com/{PAPERS_BUCKET_JSON_FILES}/jsons_from_pdfs/{SESSION_ID}/{SESSION_ID}.json"
        return render_template('diatoms_data.html', 
                             json_url=papers_json_public_url,
                             diatoms_data=DIATOMS_DATA)
    except Exception as e:
        logger.error(f"Error in diatoms_data route: {str(e)}")
        return render_template('error.html', error=str(e)), 500

@app.route('/label', methods=['GET', 'POST'])
def label():
    """Main route for the labeling interface"""
    global DIATOMS_DATA
    
    if not DIATOMS_DATA:
        try:
            DIATOMS_DATA = ClaudeAI.get_DIATOMS_DATA(PAPERS_JSON_PUBLIC_URL)
        except Exception as e:
            logger.error(f"Error loading diatoms data: {str(e)}")
            return render_template('error.html', error="No diatom data available"), 404
    
    return send_file('templates/label-react.html', mimetype='text/html')

@app.route('/api/diatoms', methods=['GET'])
def get_diatoms():
    """API endpoint to get diatom data"""
    try:
        image_index = request.args.get('index', 0, type=int)
        
        if not DIATOMS_DATA:
            try:
                DIATOMS_DATA = ClaudeAI.get_DIATOMS_DATA(PAPERS_JSON_PUBLIC_URL)
                if not DIATOMS_DATA:
                    return jsonify({
                        'current_index': 0,
                        'total_images': 0,
                        'data': {},
                        'error': 'No diatoms data available'
                    })
            except Exception as e:
                logger.error(f"Error reloading diatoms data: {str(e)}")
                return jsonify({
                    'current_index': 0,
                    'total_images': 0,
                    'data': {},
                    'error': 'Failed to load diatoms data'
                })
        
        total_images = len(DIATOMS_DATA)
        image_index = min(max(0, image_index), total_images - 1)
        
        try:
            current_image_data = DIATOMS_DATA[image_index]
            return jsonify({
                'current_index': image_index,
                'total_images': total_images,
                'data': current_image_data
            })
            
        except IndexError:
            logger.error(f"Failed to get data for index {image_index}")
            return jsonify({
                'current_index': 0,
                'total_images': total_images,
                'data': {},
                'error': 'Invalid image index'
            })
        
    except Exception as e:
        logger.error(f"Error in get_diatoms: {str(e)}")
        return jsonify({
            'current_index': 0,
            'total_images': 0,
            'data': {},
            'error': f'Error retrieving diatoms data: {str(e)}'
        }), 500

@app.route('/api/save', methods=['POST'])
def save():
    """API endpoint to save label data"""
    try:
        update_data = request.json
        success = save_labels(update_data)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Labels saved successfully',
                'timestamp': datetime.now().isoformat(),
                'saved_index': update_data.get('image_index', 0),
                'gcp_url': PAPERS_JSON_PUBLIC_URL
            })
        else:
            raise Exception("Failed to save labels")
            
    except Exception as e:
        logger.error(f"Error in save endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/download', methods=['GET'])
def download_labels():
    """Download endpoint for label data"""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(DIATOMS_DATA, temp_file, indent=4)
            temp_path = temp_file.name
        
        try:
            response = send_file(
                temp_path,
                mimetype='application/json',
                as_attachment=True,
                download_name=f'diatom_labels_{SESSION_ID}.json'
            )
            os.unlink(temp_path)
            return response
        except Exception as e:
            os.unlink(temp_path)
            raise e
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/diatom_list_assistant', methods=['GET'])
def get_diatom_list_assistant():
    """API endpoint for diatom species identification assistance"""
    try:
        image_index = request.args.get('index', 0, type=int)
        
        if not DIATOMS_DATA or image_index >= len(DIATOMS_DATA):
            return jsonify({
                'error': 'No data available or invalid index'
            }), 404

        current_image_data = DIATOMS_DATA[image_index]
        labels = [info['label'][0] for info in current_image_data.get('info', [])]
        
        pdf_text_content = ""
        matching_paper = None
        for paper in PAPER_JSON_FILES:
            if isinstance(paper.get('diatoms_data'), str):
                paper_diatoms_data = json.loads(paper['diatoms_data'])
            else:
                paper_diatoms_data = paper.get('diatoms_data', {})
                
            if paper_diatoms_data.get('image_url') == current_image_data.get('image_url'):
                pdf_text_content = paper.get('pdf_text_content', '')
                matching_paper = paper
                break

        claude = ClaudeAI()
        reformatted_labels = claude.reformat_labels_to_spaces(labels)
        messages = claude.part3_create_missing_species_prompt_and_messages(pdf_text_content, reformatted_labels)
        response = claude.get_completion(messages)

        if isinstance(response, dict) and all(k in response for k in ['species_data', 'labels_retrieved', 'message']):
            if response['species_data']:
                current_image_data['info'].extend(response['species_data'])
                
                if matching_paper:
                    if isinstance(matching_paper['diatoms_data'], str):
                        matching_paper['diatoms_data'] = json.loads(matching_paper['diatoms_data'])
                    matching_paper['diatoms_data'] = current_image_data

                    success = ClaudeAI.update_and_save_papers(
                        PAPERS_JSON_PUBLIC_URL,
                        PAPER_JSON_FILES,
                        DIATOMS_DATA
                    )
                    if not success:
                        logger.error("Failed to save updated data to GCP")

            return jsonify({
                'labels': labels,
                'pdf_text_content': pdf_text_content,
                'species_data': response.get('species_data', []),
                'labels_retrieved': response.get('labels_retrieved', []),
                'message': response.get('message', ''),
                'data_saved': bool(response['species_data'])
            })
        else:
            raise ValueError("Invalid response format from Claude")
            
    except Exception as e:
        logger.error(f"Error in diatom_list_assistant: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    # Initialize data on startup
    initialize_data()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)