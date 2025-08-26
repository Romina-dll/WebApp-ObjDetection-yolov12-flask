document.addEventListener("DOMContentLoaded", function () {
    function setupCanvas(canvasId, videoSrc) {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext("2d");
        const video = new Image();
        video.src = videoSrc;

        video.onload = function () {
            canvas.width = 540;
            canvas.height = 400;
            drawVideo();
        };

        function drawVideo() {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            requestAnimationFrame(drawVideo);
        }
    }

    // Set up first video feed
    setupCanvas("main-canvas1","/video_feed");
    //Set up the second video feed
    setupCanvas("main-canvas2", "/generate_map")

    //Line Chart Setup
    const ctxChart = document.getElementById("line-chart").getContext("2d");
    const personData = {
        labels: [],
        datasets: [{
            label: "People Count",
            borderColor: "#ff0000",
            backgroundColor: "rgba(255, 0, 0, 0.2)",
            data: [],
        }],
    };

    const lineChart = new Chart(ctxChart, {
        type: "line",
        data: personData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: "Time" } },
                y: { title: { display: true, text: "People Count" }, beginAtZero: true },
            },
        },
    });

    function updateLineChart(count) {
        const now = new Date().toLocaleTimeString();
        personData.labels.push(now);
        personData.datasets[0].data.push(count);
        if (personData.labels.length > 20) {
            personData.labels.shift();
            personData.datasets[0].data.shift();
        }
        lineChart.update();
    }
    // Fetch data every 2 seconds and update the chart
    setInterval(() => {
        fetch("/person_count")
            .then(response => response.json())
            .then(data => {
                updateLineChart(data.count);
            })
            .catch(error => console.error("Error fetching count:", error));
    }, 2000);

    // File Upload Function
    document.getElementById("fileInput").addEventListener("change", function () {
        const fileInput = this;
        if (fileInput.files.length > 0) {
            const formData = new FormData();
            formData.append("file", fileInput.files[0]);

            alert("Uploading file... Please wait.");

            fetch("/upload", {
                method: "POST",
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.message) {
                    alert("✅ File uploaded successfully!");
                    location.reload(); // Reload the entire page after successful upload
                } else {
                    alert("❌ Upload failed: " + (data.error || "Unknown error"));
                }
            })
            .catch(error => {
                console.error("Error:", error);
                alert("❌ An error occurred while uploading.");
            });
        }
    });
});
