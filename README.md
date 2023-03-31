# 3D-Rasterizer

This project implements a rasterizer in C from scratch that is capable of rendering 3D images from any arbitrary camera position using a scanline algorithm. Utilizing the Phong-shading model, Z-buffering, and linear interpolation, this program is capable of displaying objects with realistic lighting, depth, and color. 

To use this program, first, the "triangle_data.txt" file that stores triangle data (inlcuding cartesian coordinates, colors, and normal vectors for each vertex) is read in to the main program. Then, this data is processed and each triangle is rasterized with appropriate shading, lighting and depth, after which all of the image data is written to a pnm file.

The resulting video stitches together 1000 images of different camera angles into a movie that which be viewed at this link:
https://www.youtube.com/shorts/QuVRm1S6Uyk
