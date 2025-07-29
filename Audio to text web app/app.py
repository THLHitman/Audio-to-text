import os
import gc
import torch
import torchaudio
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from transformers import Wav2Vec2Processor
from importlib.machinery import SourceFileLoader
import tempfile
import threading
import time

# Configure environment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your-secret-key'

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model (loaded once)
model = None
processor = None
model_loaded = False

def load_model():
    """Load the Wav2Vec2 model and processor"""
    global model, processor, model_loaded
    try:
        print("Loading Wav2Vec2 model...")
        model_class = SourceFileLoader("model", "./wav2vec2-vi/model_handling.py").load_module().Wav2Vec2ForCTC
        model = model_class.from_pretrained("./wav2vec2-vi").to(device)
        processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-vi")
        model_loaded = True
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model_loaded = False

def convert_audio_to_wav_16k(input_path, output_path):
    """Convert any audio format to 16kHz WAV"""
    try:
        waveform, sample_rate = torchaudio.load(input_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        
        torchaudio.save(output_path, waveform, 16000)
        return True
    except Exception as e:
        print(f"Error converting audio: {e}")
        return False

def infer_by_chunks(audio_tensor, processor, model, sample_rate, chunk_sec=15):
    """Process audio in chunks to handle long files"""
    chunk_len = chunk_sec * sample_rate
    full_len = audio_tensor.shape[1]
    transcripts = []
    
    for start in range(0, full_len, chunk_len):
        end = min(start + chunk_len, full_len)
        chunk = audio_tensor[0, start:end]
        
        inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt").input_values.to(device)
        
        with torch.no_grad():
            logits = model(inputs).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        transcripts.append(transcription)
        
        # Clean up memory
        del inputs, logits, predicted_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    return " ".join(transcripts)

# Routes
@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and transcription"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded yet. Please wait and try again.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file extension
    allowed_extensions = {'mp3', 'wav', 'flac', 'm4a', 'ogg', 'wma'}
    if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return jsonify({'error': 'Unsupported file format. Please use: mp3, wav, flac, m4a, ogg, wma'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        file.save(input_path)
        
        # Convert to WAV 16kHz
        wav_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_converted.wav")
        if not convert_audio_to_wav_16k(input_path, wav_path):
            return jsonify({'error': 'Failed to convert audio file'}), 500
        
        # Load and transcribe audio
        audio, sample_rate = torchaudio.load(wav_path)
        transcription = infer_by_chunks(audio, processor, model, sample_rate)
        
        # Clean up temporary files
        os.remove(input_path)
        os.remove(wav_path)
        
        return jsonify({
            'success': True,
            'transcription': transcription,
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'error': f'Transcription failed: {str(e)}'}), 500

@app.route('/status')
def status():
    """Check model loading status"""
    return jsonify({'model_loaded': model_loaded, 'device': str(device)})

# Load model in background thread
def background_model_load():
    load_model()

if __name__ == '__main__':
    # Start model loading in background
    model_thread = threading.Thread(target=background_model_load)
    model_thread.daemon = True
    model_thread.start()
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)