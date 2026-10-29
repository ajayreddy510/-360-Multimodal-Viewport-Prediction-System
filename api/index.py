from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        html = """<!DOCTYPE html>
<html>
<head>
    <title>360° Multimodal Viewport Prediction System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0b0f19; color: #f8fafc; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
        .card { background: #111827; border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; padding: 36px; max-width: 580px; box-shadow: 0 20px 40px rgba(0,0,0,0.5); text-align: center; }
        h1 { font-size: 24px; color: #38bdf8; margin-top: 0; }
        p { color: #94a3b8; font-size: 15px; line-height: 1.6; }
        .badge { display: inline-block; padding: 6px 14px; border-radius: 9999px; background: rgba(56,189,248,0.15); color: #38bdf8; font-size: 12px; font-weight: 600; margin-bottom: 16px; }
        .links { margin-top: 24px; display: flex; flex-direction: column; gap: 10px; }
        .btn { display: block; padding: 12px 20px; border-radius: 10px; background: #f43f5e; color: white; text-decoration: none; font-weight: 600; transition: 0.2s; }
        .btn:hover { background: #e11d48; }
        .btn-sub { background: #1e293b; color: #cbd5e1; border: 1px solid rgba(255,255,255,0.08); }
        .btn-sub:hover { background: #334155; }
    </style>
</head>
<body>
    <div class="card">
        <span class="badge">🌐 SYSTEM DEPLOYMENT</span>
        <h1>360° Video Multimodal Viewport Prediction System</h1>
        <p>
            An end-to-end, timestamp-based deep learning system for predicting future head, gaze, 
            and facial movements in virtual reality playback using PyTorch Sequence GRU models.
        </p>
        <div class="links">
            <a class="btn" href="https://render.com" target="_blank">🚀 Open Streamlit Live Dashboard on Render</a>
            <a class="btn btn-sub" href="https://github.com/ajayreddy510/-360-Multimodal-Viewport-Prediction-System" target="_blank">📂 View GitHub Repository</a>
        </div>
    </div>
</body>
</html>"""
        self.wfile.write(html.encode('utf-8'))
        return
