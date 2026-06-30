This is a beta version of a simplified program that uses OpenCV to identify parameters such as the wall thickness and size of bamboo.
It can run on CPUs with very weak performance, with no need for NPU, GPU, etc.
In real-world applications, additional equipment is required,
such as optical calibration boards, line lasers, servo motors, development boards, and so on.
When you have access to better-performing hardware—for example,
if your development board includes an NPU—it is strongly recommended that you use a YOLO model instead of the original OpenCV approach in this simple program,
as YOLO models offer significantly higher accuracy.
