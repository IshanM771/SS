from flask import Flask, render_template, request, send_file
#from face_datasets import collect_training_data
from loggen import face_recognition_from_video

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/service1", methods=['POST'])
def service1():
    if request.method == "POST":
        name = request.form.get('name')
        if name:
            collect_training_data(name)
            message = f"Training data collected for {name}."
            return render_template("service1.html", message=message)
        else:
            error = "Name field is required."
            return render_template("service1.html", error=error)
    return render_template("service1.html")

@app.route("/service2", methods=['POST'])
def service2():
    if request.method == "POST":
        video_file = request.files['filename']
        if video_file.filename:
            video_file.save(video_file.filename)
            log_file_path = face_recognition_from_video(video_file.filename)
            return render_template("service2.html", result="Face recognition performed successfully.", log_file=log_file_path)
        else:
            error = "Filename field is required."
            return render_template("service2.html", error=error)
    return render_template("service2.html")

@app.route("/log_file.txt")
def download_log_file():
    log_file_path = request.args.get('log_file')
    return send_file(log_file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
