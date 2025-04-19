let mediaRecorder;
let audioChunks = [];
let audioFile = null;
let recordingTimer = null;
let secondsElapsed = 0;

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        audioFile = file;
        stopTimer();
        document.getElementById("result").innerText = "📁 Audio file ready for identification.";
    }
}

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            secondsElapsed = 0;

            mediaRecorder.ondataavailable = event => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                audioFile = new File([audioBlob], "recording.wav", { type: 'audio/wav' });
                document.getElementById("result").innerText = "🎙️ Recording complete. Ready for identification.";
                stopTimer();
            };

            mediaRecorder.start();
            document.getElementById("result").innerText = "🔴 Recording... 0s";
            startTimer();

            // Stop recording automatically after 2 minutes
            setTimeout(() => {
                if (mediaRecorder && mediaRecorder.state === "recording") {
                    mediaRecorder.stop();
                    document.getElementById("result").innerText += "\n⏱️ Max time reached. Recording stopped.";
                }
            }, 2 * 60 * 1000);
        })
        .catch(error => {
            console.error("Microphone access denied or not available:", error);
            alert("Microphone access is required for recording.");
        });
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        stopTimer();
    }
}

function startTimer() {
    recordingTimer = setInterval(() => {
        secondsElapsed++;
        document.getElementById("result").innerText = `🔴 Recording... ${secondsElapsed}s`;

        if (secondsElapsed >= 120) {
            stopTimer();
        }
    }, 1000);
}

function stopTimer() {
    clearInterval(recordingTimer);
    recordingTimer = null;
    secondsElapsed = 0;
}

async function identifyVoice() {
    if (!audioFile) {
        alert("Please upload or record an audio file first.");
        return;
    }

    document.getElementById("result").innerText = "🔄 Processing audio... Please wait.";

    const formData = new FormData();
    formData.append("audio", audioFile);

    try {
        const response = await fetch("http://localhost:5000/predict", {
            method: "POST",
            body: formData
        });

        const result = await response.json();
        if (result.speaker) {
            document.getElementById("result").innerText = "🧠 Identified Speaker: " + result.speaker;
        } else {
            document.getElementById("result").innerText = "⚠️ Unable to identify speaker.";
        }
    } catch (error) {
        document.getElementById("result").innerText = "❌ Error identifying voice.";
        console.error("Prediction error:", error);
    }
}
