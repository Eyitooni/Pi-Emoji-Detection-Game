"""
web_app.py
Serves the live camera view and expression analysis as a web page.

Run:    python web_app.py
Open:   http://<pi-address>:5000   (from any device on your network)
"""

import argparse
import os
import socket

from flask import Flask, Response, jsonify, send_file, send_from_directory

from expression_engine import ExpressionEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_app(engine):
    app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"))

    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/video")
    def video():
        def stream():
            seq = 0
            while engine.running:
                new_seq, jpeg = engine.wait_jpeg(seq, timeout=2.0)
                if new_seq == seq or jpeg is None:
                    continue
                seq = new_seq
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")

        return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/state")
    def state():
        resp = jsonify(engine.get_state())
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.route("/api/snapshot")
    def snapshot():
        path = engine.save_snapshot()
        if not path:
            return jsonify(error="No frame yet"), 503
        return send_file(path, as_attachment=True)

    @app.route("/api/reset", methods=["POST"])
    def reset():
        engine.reset_session()
        return jsonify(ok=True)

    @app.route("/api/mirror", methods=["POST"])
    def mirror():
        engine.mirror = not engine.mirror
        return jsonify(mirror=engine.mirror)

    @app.route("/api/mirror", methods=["GET"])
    def mirror_state():
        return jsonify(mirror=engine.mirror)

    return app


def serve(engine, host="0.0.0.0", port=5000):
    app = create_app(engine)
    ip = "localhost"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
    except OSError:
        pass
    print("\nOpen in a browser:  http://%s:%d\n" % (ip, port))
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expression monitor web interface")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    engine = ExpressionEngine()
    engine.start()
    try:
        serve(engine, args.host, args.port)
    finally:
        engine.stop()
