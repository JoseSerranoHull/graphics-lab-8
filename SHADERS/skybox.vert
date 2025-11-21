#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
	mat4 model;
	mat4 view;
	mat4 proj;
	vec3 eyePos;
} ubo;

layout(location = 0) in vec3 inPosition;

layout(location = 1) out vec3 viewDir;

void main() {
	// viewDir is the direction vector for cubemap sampling
	viewDir = inPosition;

	// Use only the rotational component of the view matrix (mat3) 
    // to remove camera translation, keeping the skybox centered.
	mat4 rotView = mat4(mat3(ubo.view));

	// Calculate clip position
	vec4 clipPos = ubo.proj * rotView * vec4(inPosition, 1.0);

	// The trick: Set gl_Position.w = gl_Position.z to force z/w = 1.0 after perspective divide.
    // This makes the skybox appear at the maximum depth (far plane).
	gl_Position = clipPos.xyww;
}