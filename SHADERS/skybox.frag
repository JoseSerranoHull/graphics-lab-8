#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 1) in vec3 viewDir; // Must match location from vert shader

layout(binding = 1) uniform samplerCube skySampler;

layout(location = 0) out vec4 outColor;

void main() {
	// Sample the cubemap using the vertex's position as a 3D direction
	outColor = texture(skySampler, viewDir);
}