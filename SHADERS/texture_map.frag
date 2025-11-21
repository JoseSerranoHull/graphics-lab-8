#version 450

layout(location = 1) in vec2 fragTexCoord;

// This binding will point to our 'offscreenImage'
layout(binding = 1) uniform sampler2D sceneTexture;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(sceneTexture, fragTexCoord);
}