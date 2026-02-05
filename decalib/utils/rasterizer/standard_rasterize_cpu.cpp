#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

// Helper struct for 2D points
template <typename T>
struct Point {
    T x, y;
};

// Barycentric weight calculation
template <typename T>
void barycentric_weight(T* w, Point<T> p, Point<T> p0, Point<T> p1, Point<T> p2) {
    Point<T> v0 = {p2.x - p0.x, p2.y - p0.y};
    Point<T> v1 = {p1.x - p0.x, p1.y - p0.y};
    Point<T> v2 = {p.x - p0.x, p.y - p0.y};

    T dot00 = v0.x * v0.x + v0.y * v0.y;
    T dot01 = v0.x * v1.x + v0.y * v1.y;
    T dot02 = v0.x * v2.x + v0.y * v2.y;
    T dot11 = v1.x * v1.x + v1.y * v1.y;
    T dot12 = v1.x * v2.x + v1.y * v2.y;

    T denom = dot00 * dot11 - dot01 * dot01;
    T invDenom = (denom == 0) ? 0 : 1 / denom;

    T u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    T v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    w[0] = 1 - u - v;
    w[1] = v;
    w[2] = u;
}

void standard_rasterize_cpu_kernel(
    const torch::Tensor& face_vertices,
    torch::Tensor& depth_buffer,
    torch::Tensor& triangle_buffer,
    torch::Tensor& baryw_buffer,
    int h, int w
) {
    // face_vertices: [B, F, 3, 3]
    // depth_buffer: [B, H, W]
    // triangle_buffer: [B, H, W]
    // baryw_buffer: [B, H, W, 3]

    auto B = face_vertices.size(0);
    auto F = face_vertices.size(1);

    // Get accessors for efficient element access
    // Assuming float input as per standard renderer usage
    auto face_vertices_a = face_vertices.accessor<float, 4>();
    auto depth_buffer_a = depth_buffer.accessor<float, 3>();
    auto triangle_buffer_a = triangle_buffer.accessor<int, 3>();
    auto baryw_buffer_a = baryw_buffer.accessor<float, 4>();

    for (int b = 0; b < B; ++b) {
        for (int f = 0; f < F; ++f) {
            float p0x = face_vertices_a[b][f][0][0];
            float p0y = face_vertices_a[b][f][0][1];
            float p0z = face_vertices_a[b][f][0][2];

            float p1x = face_vertices_a[b][f][1][0];
            float p1y = face_vertices_a[b][f][1][1];
            float p1z = face_vertices_a[b][f][1][2];

            float p2x = face_vertices_a[b][f][2][0];
            float p2y = face_vertices_a[b][f][2][1];
            float p2z = face_vertices_a[b][f][2][2];

            // Calculate bounding box for the triangle
            int x_min = std::max(0, (int)std::ceil(std::min({p0x, p1x, p2x})));
            int x_max = std::min(w - 1, (int)std::floor(std::max({p0x, p1x, p2x})));
            int y_min = std::max(0, (int)std::ceil(std::min({p0y, p1y, p2y})));
            int y_max = std::min(h - 1, (int)std::floor(std::max({p0y, p1y, p2y})));

            if (x_min > x_max || y_min > y_max) continue;

            Point<float> p0 = {p0x, p0y};
            Point<float> p1 = {p1x, p1y};
            Point<float> p2 = {p2x, p2y};

            for (int y = y_min; y <= y_max; ++y) {
                for (int x = x_min; x <= x_max; ++x) {
                    Point<float> p = {(float)x, (float)y};
                    float bw[3];
                    barycentric_weight(bw, p, p0, p1, p2);

                    // Check if pixel is inside the triangle
                    if (bw[0] > 0 && bw[1] >= 0 && bw[2] >= 0) {
                        // Perspective correct depth interpolation
                        float zp = 1.0f / (bw[0] / p0z + bw[1] / p1z + bw[2] / p2z);
                        
                        // Z-buffer test
                        if (zp < depth_buffer_a[b][y][x]) {
                            depth_buffer_a[b][y][x] = zp;
                            triangle_buffer_a[b][y][x] = f;
                            baryw_buffer_a[b][y][x][0] = bw[0];
                            baryw_buffer_a[b][y][x][1] = bw[1];
                            baryw_buffer_a[b][y][x][2] = bw[2];
                        }
                    }
                }
            }
        }
    }
}

void standard_rasterize(
    torch::Tensor face_vertices,
    torch::Tensor depth_buffer,
    torch::Tensor triangle_buffer,
    torch::Tensor baryw_buffer,
    int h, int w
) {
    standard_rasterize_cpu_kernel(face_vertices, depth_buffer, triangle_buffer, baryw_buffer, h, w);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("standard_rasterize", &standard_rasterize, "Standard Rasterize (CPU)");
}
