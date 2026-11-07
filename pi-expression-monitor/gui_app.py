"""
gui_app.py
Desktop (Tkinter) interface for live face detection and expression analysis.

Run:            python gui_app.py
Run with web:   python gui_app.py --web      (also serves the web page on port 5000)

The camera can only be used by one program at a time, so use --web if you
want the desktop window and the web page together.
"""

import argparse
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from expression_engine import COLORS_HEX, EXPRESSIONS, ExpressionEngine

LABELS = {
    "happy": "Happy", "surprised": "Surprised", "sad": "Sad",
    "angry": "Angry", "disgust": "Disgusted", "neutral": "Neutral",
}
VIDEO_W, VIDEO_H = 800, 450


class App:
    def __init__(self, root, engine):
        self.root = root
        self.engine = engine
        self.photo = None

        root.title("Expression monitor")
        root.configure(bg="#EEF1F5")
        root.protocol("WM_DELETE_WINDOW", self.close)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#FFFFFF")
        style.configure("TLabel", background="#FFFFFF", foreground="#16202B", font=("Helvetica", 11))
        style.configure("Muted.TLabel", foreground="#5E6A79")
        style.configure("Head.TLabel", foreground="#5E6A79", font=("Helvetica", 10, "bold"))
        style.configure("Big.TLabel", font=("Helvetica", 26, "bold"))
        for e in EXPRESSIONS:
            style.configure("%s.Horizontal.TProgressbar" % e,
                            background=COLORS_HEX[e], troughcolor="#E6EAF0",
                            bordercolor="#E6EAF0", lightcolor=COLORS_HEX[e],
                            darkcolor=COLORS_HEX[e])

        # ---- left: video
        left = tk.Frame(root, bg="#EEF1F5")
        left.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="n")
        self.video = tk.Label(left, bg="#0C1116", width=VIDEO_W, height=VIDEO_H,
                              image=self._blank(), compound="center")
        self.video.pack()

        buttons = tk.Frame(left, bg="#EEF1F5")
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="Save snapshot", command=self.snapshot).pack(side="left")
        ttk.Button(buttons, text="Reset session", command=self.engine.reset_session).pack(side="left", padx=8)
        self.mirror_var = tk.BooleanVar(value=engine.mirror)
        ttk.Checkbutton(buttons, text="Mirror view", variable=self.mirror_var,
                        command=self.toggle_mirror).pack(side="left")
        ttk.Button(buttons, text="Quit", command=self.close).pack(side="right")

        # ---- right: analysis panel
        panel = ttk.Frame(root, padding=20)
        panel.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="n")

        ttk.Label(panel, text="RIGHT NOW", style="Head.TLabel").pack(anchor="w")
        now = tk.Frame(panel, bg="#FFFFFF")
        now.pack(fill="x", pady=(6, 0))
        self.swatch = tk.Frame(now, width=6, bg=COLORS_HEX["neutral"])
        self.swatch.pack(side="left", fill="y", padx=(0, 12))
        text = tk.Frame(now, bg="#FFFFFF")
        text.pack(side="left")
        self.now_label = ttk.Label(text, text="Waiting for a face", style="Big.TLabel")
        self.now_label.pack(anchor="w")
        self.now_sub = ttk.Label(text, text="Face the camera in even light.", style="Muted.TLabel")
        self.now_sub.pack(anchor="w")

        ttk.Label(panel, text="EXPRESSION MIX", style="Head.TLabel").pack(anchor="w", pady=(20, 4))
        self.bars, self.pcts = {}, {}
        for e in EXPRESSIONS:
            row = tk.Frame(panel, bg="#FFFFFF")
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=LABELS[e], width=10).pack(side="left")
            bar = ttk.Progressbar(row, length=170, maximum=100,
                                  style="%s.Horizontal.TProgressbar" % e)
            bar.pack(side="left", padx=6)
            pct = ttk.Label(row, text="0%", width=5, style="Muted.TLabel", anchor="e")
            pct.pack(side="left")
            self.bars[e], self.pcts[e] = bar, pct

        ttk.Label(panel, text="FACE DETAILS", style="Head.TLabel").pack(anchor="w", pady=(20, 4))
        self.details = {}
        for key in ("Faces in view", "Eyes", "Mouth", "Gaze", "Frame rate"):
            row = tk.Frame(panel, bg="#FFFFFF")
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=key, style="Muted.TLabel").pack(side="left")
            val = ttk.Label(row, text="-")
            val.pack(side="right")
            self.details[key] = val

        ttk.Label(panel, text="THIS SESSION", style="Head.TLabel").pack(anchor="w", pady=(20, 4))
        self.session_label = ttk.Label(panel, text="No data yet", style="Muted.TLabel",
                                       wraplength=290, justify="left")
        self.session_label.pack(anchor="w")

        self.update()

    @staticmethod
    def _blank():
        return ImageTk.PhotoImage(Image.new("RGB", (VIDEO_W, VIDEO_H), "#0C1116"))

    # ---- actions
    def toggle_mirror(self):
        self.engine.mirror = self.mirror_var.get()

    def snapshot(self):
        path = self.engine.save_snapshot()
        if path:
            messagebox.showinfo("Snapshot saved", path)

    def close(self):
        self.engine.stop()
        self.root.destroy()

    # ---- periodic refresh
    def update(self):
        frame = self.engine.get_frame()
        if frame is not None:
            frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video.configure(image=self.photo)

        s = self.engine.get_state()
        if s.get("error"):
            self.now_label.configure(text="Camera problem")
            self.now_sub.configure(text=s["error"])
        else:
            self.render(s)
        self.root.after(33, self.update)

    def render(self, s):
        face = None if s["primary"] is None else s["faces"][s["primary"]]
        self.details["Faces in view"].configure(text=str(s["face_count"]))
        self.details["Frame rate"].configure(text="%d fps" % round(s["fps"]))

        if face:
            self.now_label.configure(text=LABELS[face["expression"]])
            self.now_sub.configure(text="%d%% confident" % round(face["confidence"] * 100))
            self.swatch.configure(bg=COLORS_HEX[face["expression"]])
            self.details["Eyes"].configure(text=face["eyes"])
            self.details["Mouth"].configure(text=face["mouth"])
            self.details["Gaze"].configure(text=face["gaze"])
        else:
            self.now_label.configure(text="Waiting for a face")
            self.now_sub.configure(text="Face the camera in even light.")
            self.swatch.configure(bg=COLORS_HEX["neutral"])
            for k in ("Eyes", "Mouth", "Gaze"):
                self.details[k].configure(text="-")

        for e in EXPRESSIONS:
            v = face["scores"][e] if face else 0.0
            self.bars[e]["value"] = v * 100
            self.pcts[e].configure(text="%d%%" % round(v * 100))

        parts = ["%s %d%%" % (LABELS[e], round(p * 100))
                 for e, p in sorted(s["session"].items(), key=lambda kv: -kv[1]) if p > 0]
        self.session_label.configure(text=",  ".join(parts) if parts else "No data yet")


def main():
    parser = argparse.ArgumentParser(description="Expression monitor desktop GUI")
    parser.add_argument("--web", action="store_true", help="also serve the web page")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    engine = ExpressionEngine(size=(960, 540))
    engine.start()

    if args.web:
        from web_app import serve
        threading.Thread(target=serve, args=(engine, "0.0.0.0", args.port), daemon=True).start()

    root = tk.Tk()
    App(root, engine)
    root.mainloop()


if __name__ == "__main__":
    main()
