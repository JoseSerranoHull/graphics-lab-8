#version 450

layout(location = 1) in vec2 fragTexCoord;

// This binding points to our 'offscreenImage' (Pass 1 output)
layout(binding = 1) uniform sampler2D sceneTexture;

layout(location = 0) out vec4 outColor;

void main() {
    // 1. Calculate size of a single texel
    // textureSize returns the width/height of the image in pixels
    // stepSize allows us to jump further to make the blur stronger (e.g., 1.0, 2.0, 3.0)
    float stepSize = 3.0; 
    vec2 texelSize = stepSize / vec2(textureSize(sceneTexture, 0));
    
    vec4 result = vec4(0.0);
    
    // 2. Convolution Loop (Box Blur)
    // We sample a grid around the center pixel
    // boxSize = 2 means a 5x5 grid (-2 to +2)
    int boxSize = 6; 

    for (int x = -boxSize; x <= boxSize; x++) {
        for (int y = -boxSize; y <= boxSize; y++) {
            // Offset coordinate by (x, y) texels
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            result += texture(sceneTexture, fragTexCoord + offset);
        }
    }

    // 3. Average the result
    // Total samples = width * height of the grid
    int totalSamples = (boxSize * 2 + 1) * (boxSize * 2 + 1);
    
    outColor = result / float(totalSamples);
}