#include "GeometryUtils.h"
#include <glm/gtc/noise.hpp>
#include <cmath>
#include <glm/ext/scalar_constants.hpp> // For the glm::pi<float>()

GeometryUtils::MeshData GeometryUtils::CreateCube(float size, const glm::vec3& offset) {
    MeshData mesh;

    float hs = size * 0.5f;

    // Define cube faces
    struct Face {
        glm::vec3 normal;
        glm::vec3 color;
        glm::vec3 v[4];
    };

    /* NORMAL CUBE*/
    
    std::vector<Face> faces = {
        // +X
        { { 1, 0, 0 }, {1, 0, 0}, { {hs, -hs, -hs}, {hs, hs, -hs}, {hs, hs, hs}, {hs, -hs, hs} } },
        // -X
        { {-1, 0, 0 }, {0, 1, 0}, { {-hs, -hs, hs}, {-hs, hs, hs}, {-hs, hs, -hs}, {-hs, -hs, -hs} } },
        // +Y
        { { 0, 1, 0 }, {0, 0, 1}, { {-hs, hs, -hs}, {-hs, hs, hs}, {hs, hs, hs}, {hs, hs, -hs} } },
        // -Y
        { { 0,-1, 0 }, {1, 1, 0}, { {-hs, -hs, hs}, {-hs, -hs, -hs}, {hs, -hs, -hs}, {hs, -hs, hs} } },
        // +Z
        { { 0, 0, 1 }, {0, 1, 1}, { {-hs, -hs, hs}, {hs, -hs, hs}, {hs, hs, hs}, {-hs, hs, hs} } },
        // -Z
        { { 0, 0,-1 }, {1, 0, 1}, { {hs, -hs, -hs}, {-hs, -hs, -hs}, {-hs, hs, -hs}, {hs, hs, -hs} } },
    };

    glm::vec2 tex[4] = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f},
        {0.0f, 1.0f}
    };

    for (const auto& f : faces) {
        uint32_t startIndex = static_cast<uint32_t>(mesh.vertices.size());

        for (int i = 0; i < 4; i++) {
            GeometryVertex v{};
            v.pos = f.v[i];
            v.color = f.color;
            v.normal = f.normal;
            v.texCoord = tex[i];
            mesh.vertices.push_back(v);
        }

        // For each face in CreateCube:
        glm::vec3 edge1 = f.v[1] - f.v[0];
        glm::vec3 edge2 = f.v[2] - f.v[0];
        glm::vec2 deltaUV1 = tex[1] - tex[0];
        glm::vec2 deltaUV2 = tex[2] - tex[0];

        float f_denom = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
        glm::vec3 tangent = f_denom * (deltaUV2.y * edge1 - deltaUV1.y * edge2);
        glm::vec3 binormal = f_denom * (-deltaUV2.x * edge1 + deltaUV1.x * edge2);

        tangent = glm::normalize(tangent);
        binormal = glm::normalize(binormal);

        // Then, for each vertex:
        for (int i = 0; i < 4; i++) {
            mesh.vertices[startIndex + i].tangent = tangent;
            mesh.vertices[startIndex + i].binormal = binormal;
        }

        // Two triangles per face
        mesh.indices.push_back(startIndex + 0);
        mesh.indices.push_back(startIndex + 1);
        mesh.indices.push_back(startIndex + 2);
        mesh.indices.push_back(startIndex + 2);
        mesh.indices.push_back(startIndex + 3);
        mesh.indices.push_back(startIndex + 0);
    }

    return mesh;
}

GeometryUtils::MeshData GeometryUtils::CreateGrid(int width, int depth, const glm::vec3& offset) {
    MeshData mesh;

    // Generate vertices
    for (int z = 0; z <= depth; ++z) {
        for (int x = 0; x <= width; ++x) {
            float fx = static_cast<float>(x) - width / 2.0f;
            float fz = static_cast<float>(z) - depth / 2.0f;
            glm::vec3 pos = glm::vec3(fx, 0.0f, fz) + offset;
            mesh.vertices.push_back(GeometryVertex{
                pos,
                glm::vec3(fx / width + 0.5f, fz / depth + 0.5f, 1.0f)
                });
        }
    }

    // Generate indices for triangle strips with primitive restart
    const uint32_t RESTART_INDEX = 0xFFFFFFFF;
    for (int z = 0; z < depth; ++z) {
        for (int x = 0; x <= width; ++x) {
            mesh.indices.push_back(z * (width + 1) + x);
            mesh.indices.push_back((z + 1) * (width + 1) + x);
        }
        if (z < depth - 1) {
            mesh.indices.push_back(RESTART_INDEX);
        }
    }

    return mesh;
}

GeometryUtils::MeshData GeometryUtils::CreateTerrain(int width, int depth, float scale, const glm::vec3& offset) {
    MeshData mesh;

    // Generate vertices with Perlin noise height
    for (int z = 0; z <= depth; ++z) {
        for (int x = 0; x <= width; ++x) {
            float fx = static_cast<float>(x) - width / 2.0f;
            float fz = static_cast<float>(z) - depth / 2.0f;
            float fy = glm::perlin(glm::vec2(fx * scale, fz * scale)) * 2.0f; // Controls amplitude
            glm::vec3 pos = glm::vec3(fx, fy, fz) + offset;
            mesh.vertices.push_back(GeometryVertex{
                pos,
                glm::vec3(fx / width + 0.5f, fy, fz / depth + 0.5f)
                });
        }
    }

    // Generate indices for triangle strips with primitive restart
    const uint32_t RESTART_INDEX = 0xFFFFFFFF;
    for (int z = 0; z < depth; ++z) {
        for (int x = 0; x <= width; ++x) {
            mesh.indices.push_back(z * (width + 1) + x);
            mesh.indices.push_back((z + 1) * (width + 1) + x);
        }
        if (z < depth - 1) {
            mesh.indices.push_back(RESTART_INDEX);
        }
    }

    return mesh;
}

GeometryUtils::MeshData GeometryUtils::CreateCylinder(float radius, float height, int segments, const glm::vec3& offset) {
    MeshData mesh;

    // --- Generate bottom and top circle vertices ---
    // Center vertices
    uint32_t bottomCenterIndex = 0;
    uint32_t topCenterIndex = 1;
    mesh.vertices.push_back(GeometryVertex{ glm::vec3(0.0f, 0.0f, 0.0f) + offset, glm::vec3(1.0f, 0.0f, 0.0f) }); // bottom center
    mesh.vertices.push_back(GeometryVertex{ glm::vec3(0.0f, height, 0.0f) + offset, glm::vec3(0.0f, 1.0f, 0.0f) }); // top center

    // Outer circle vertices
    for (int i = 0; i < segments; ++i) {
        float theta = 2.0f * glm::pi<float>() * i / segments;
        float x = radius * std::cos(theta);
        float z = radius * std::sin(theta);

        // Bottom circle
        mesh.vertices.push_back(GeometryVertex{ glm::vec3(x, 0.0f, z) + offset, glm::vec3(1.0f, 0.0f, 0.0f) });
        // Top circle
        mesh.vertices.push_back(GeometryVertex{ glm::vec3(x, height, z) + offset, glm::vec3(0.0f, 1.0f, 0.0f) });
    }

    // --- Indices for bottom cap ---
    for (int i = 0; i < segments; ++i) {
        uint32_t next = (i + 1) % segments;
        mesh.indices.push_back(bottomCenterIndex);
        mesh.indices.push_back(2 + i * 2);
        mesh.indices.push_back(2 + next * 2);
    }

    // --- Indices for top cap ---
    for (int i = 0; i < segments; ++i) {
        uint32_t next = (i + 1) % segments;
        mesh.indices.push_back(topCenterIndex);
        mesh.indices.push_back(3 + i * 2);
        mesh.indices.push_back(3 + next * 2);
    }

    // --- Indices for cylinder wall ---
    for (int i = 0; i < segments; ++i) {
        uint32_t b0 = 2 + i * 2;
        uint32_t t0 = b0 + 1;
        uint32_t b1 = 2 + ((i + 1) % segments) * 2;
        uint32_t t1 = b1 + 1;

        // First triangle
        mesh.indices.push_back(b0);
        mesh.indices.push_back(t0);
        mesh.indices.push_back(b1);

        // Second triangle
        mesh.indices.push_back(b1);
        mesh.indices.push_back(t0);
        mesh.indices.push_back(t1);
    }

    return mesh;
}

GeometryUtils::MeshData GeometryUtils::CreateSphere(float radius, int stacks, int slices, const glm::vec3& offset) {
    MeshData mesh;
    // Generate vertices
    for (int i = 0; i <= stacks; ++i) {
        float phi = glm::pi<float>() * i / stacks; // [0, pi]
        float y = radius * std::cos(phi);
        float r = radius * std::sin(phi);
        for (int j = 0; j <= slices; ++j) {
            float theta = 2.0f * glm::pi<float>() * j / slices; // [0, 2pi]
            float x = r * std::cos(theta);
            float z = r * std::sin(theta);
            mesh.vertices.push_back(GeometryVertex{
                glm::vec3(x, y, z) + offset,
                glm::vec3((float)j / slices, (float)i / stacks, 1.0f)
                });
        }
    }
    // Generate indices
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            int first = i * (slices + 1) + j;
            int second = first + slices + 1;
            mesh.indices.push_back(first);
            mesh.indices.push_back(second);
            mesh.indices.push_back(first + 1);

            mesh.indices.push_back(second);
            mesh.indices.push_back(second + 1);
            mesh.indices.push_back(first + 1);
        }
    }

    return mesh;
}

GeometryUtils::MeshData GeometryUtils::CreateQuad(float width, float height, const glm::vec3& offset) {
    MeshData mesh;

    float hw = width * 0.5f;
    float hh = height * 0.5f;

    // Vertex positions and texture coordinates
    glm::vec3 positions[4] = {
        glm::vec3(-hw, -hh, 0.0f) + offset, // bottom-left
        glm::vec3(hw, -hh, 0.0f) + offset,  // bottom-right
        glm::vec3(hw,  hh, 0.0f) + offset,  // top-right
        glm::vec3(-hw,  hh, 0.0f) + offset  // top-left
    };
    glm::vec2 texCoords[4] = {
        glm::vec2(0, 1),
        glm::vec2(1, 1),
        glm::vec2(1, 0),
        glm::vec2(0, 0)
    };
    glm::vec3 colors[4] = {
        glm::vec3(1,0,0),
        glm::vec3(0,1,0),
        glm::vec3(0,0,1),
        glm::vec3(1,1,1)
    };

    // Calculate tangent and binormal using the first triangle (0,1,2)
    glm::vec3 edge1 = positions[1] - positions[0];
    glm::vec3 edge2 = positions[2] - positions[0];
    glm::vec2 deltaUV1 = texCoords[1] - texCoords[0];
    glm::vec2 deltaUV2 = texCoords[2] - texCoords[0];

    float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

    glm::vec3 tangent = f * (deltaUV2.y * edge1 - deltaUV1.y * edge2);
    glm::vec3 binormal = f * (-deltaUV2.x * edge1 + deltaUV1.x * edge2);

    tangent = glm::normalize(tangent);
    binormal = glm::normalize(binormal);
    glm::vec3 normal = glm::normalize(glm::cross(tangent, binormal));

    // Assign to all vertices (for a flat quad, all are the same)
    for (int i = 0; i < 4; ++i) {
        mesh.vertices.push_back(GeometryVertex{
            positions[i],
            colors[i],
            normal,
            texCoords[i],
            tangent,
            binormal
            });
    }

    // Indices for two triangles
    mesh.indices = { 0, 1, 2, 2, 3, 0 };

    return mesh;
}