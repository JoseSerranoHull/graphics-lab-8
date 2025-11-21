#version 450

layout(location = 1) in vec2 fragTexCoord;
layout(binding = 1) uniform sampler2D sceneTexture;
layout(location = 0) out vec4 outColor;

void main() {
    // 1. Get Original
    vec4 originalColor = texture(sceneTexture, fragTexCoord);

    // 2. Box Blur
    // INCREASED stepSize from 3.0 to 6.0 to make the blur spread wider
    float stepSize = 6.0; 
    vec2 texelSize = stepSize / vec2(textureSize(sceneTexture, 0));
    
    vec4 blurredColor = vec4(0.0);
    int boxSize = 6; 

    for (int x = -boxSize; x <= boxSize; x++) {
        for (int y = -boxSize; y <= boxSize; y++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            blurredColor += texture(sceneTexture, fragTexCoord + offset);
        }
    }

    int totalSamples = (boxSize * 2 + 1) * (boxSize * 2 + 1);
    blurredColor = blurredColor / float(totalSamples);

    // 3. Additive Blending (Glow)
    // INCREASED intensity from 0.8 to 2.0 to make it pop
    outColor = originalColor + (blurredColor * 2.0);
}