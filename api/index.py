from http.server import BaseHTTPRequestHandler
from pathlib import Path

PUBLIC_DIR = Path(__file__).resolve().parent.parent / "public"
INDEX_FILE = PUBLIC_DIR / "index.html"

FALLBACK_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>360° Multimodal Viewport Prediction System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: sans-serif; background: #080c14; color: #fff; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
        .card { background: #0f172a; padding: 40px; border-radius: 16px; text-align: center; max-width: 600px; border: 1px solid rgba(255,255,255,0.1); }
        h1 { color: #38bdf8; font-size: 24px; }
        p { color: #94a3b8; line-height: 1.6; }
        a { display: inline-block; margin-top: 16px; padding: 12px 24px; background: #f43f5e; color: #fff; text-decoration: none; border-radius: 8px; font-weight: 600; }
    </style>
</head>
<body>
    <div class="card">
        <h1>360° Multimodal Viewport Prediction System</h1>
        <p>AI Deep Sequence Architecture for real-time viewport and trajectory prediction in immersive 360° virtual reality.</p>
        <a href="https://360-multimodal-viewport-prediction-system.onrender.com" target="_blank">Launch Streamlit Live App on Render</a>
    </div>
</body>
</html>"""

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        
        if INDEX_FILE.exists():
            try:
                content = INDEX_FILE.read_bytes()
            except Exception:
                content = FALLBACK_HTML.encode('utf-8')
        else:
            content = FALLBACK_HTML.encode('utf-8')
            
        self.wfile.write(content)
        return
