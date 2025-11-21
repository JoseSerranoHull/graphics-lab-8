#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <vulkan/vulkan_core.h>
#include <array>

struct GeometryVertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec3 tangent;
    glm::vec3 binormal;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(GeometryVertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 6> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 6> attributeDescriptions{};
        attributeDescriptions[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(GeometryVertex, pos) };
        attributeDescriptions[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(GeometryVertex, color) };
        attributeDescriptions[2] = { 2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(GeometryVertex, normal) };
        attributeDescriptions[3] = { 3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(GeometryVertex, texCoord) };
        attributeDescriptions[4] = { 4, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(GeometryVertex, tangent) };
        attributeDescriptions[5] = { 5, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(GeometryVertex, binormal) };
        return attributeDescriptions;
    }

    static VkVertexInputAttributeDescription getTexCoordAttributeDescription() {
        VkVertexInputAttributeDescription texCoordDescription{};
        texCoordDescription.binding = 0;
        texCoordDescription.location = 3;
        texCoordDescription.format = VK_FORMAT_R32G32_SFLOAT;
        texCoordDescription.offset = offsetof(GeometryVertex, texCoord);
        return texCoordDescription;
    }
};

namespace GeometryUtils {

    struct MeshData {
        std::vector<GeometryVertex> vertices;
        std::vector<uint32_t> indices;
    };

	MeshData CreateCube(float size, const glm::vec3& offset);
    MeshData CreateGrid(int width, int depth, const glm::vec3& offset);
    MeshData CreateTerrain(int width, int depth, float scale, const glm::vec3& offset);
    MeshData CreateCylinder(float radius, float height, int segments, const glm::vec3& offset);
    MeshData CreateSphere(float radius, int stacks, int slices, const glm::vec3& offset);
    MeshData CreateQuad(float width, float height, const glm::vec3& offset);

}