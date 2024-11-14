Current solution:

Step 1: Get the RTSP stream

Step 2: Process each frame in the stream

Step 3: Transform the processed frame into a MJPEG stream

Step 4: Display the MJPEG stream on the browser

To run the code, type python hm.py and open 127.0.0.1:3000 in a browser. You should be able to see the processed stream.
For Mathew's code, type python mathew.py and open 127.0.0.1:5000 in a browser.

The environment uses PyTorch version 1.13.1 with CUDA 11.6 (cu116), and Python version 3.9.18.

You need to download yolov7-e6e.pt from https://github.com/WongKinYiu/yolov7 and place it in the weights folder, as the file size exceeds GitHub's upload limit.