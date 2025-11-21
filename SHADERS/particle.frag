#version 450

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in float fragT; // Lifecycle 0.0 (bottom) to 1.0 (top)

layout(location = 0) out vec4 outColor;

void main() {
    // 1. Circular shape logic
    vec2 coord = fragTexCoord * 2.0 - 1.0;
    float dist = length(coord);
    if (dist > 1.0) discard;

    // 2. Define Color Palette
    vec3 colorWhite  = vec3(1.0, 1.0, 1.0); // Hottest base
    vec3 colorYellow = vec3(1.0, 0.8, 0.1); // Body
    vec3 colorRed    = vec3(1.0, 0.1, 0.0); // Top flames
    vec3 colorGrey   = vec3(0.2, 0.2, 0.2); // Smoke

    // 3. Calculate color based on Vertical Lifecycle (fragT)
    vec3 finalColor;
    
    if (fragT < 0.3) {
        // Bottom: Mix White -> Yellow
        finalColor = mix(colorWhite, colorYellow, fragT / 0.3);
    } else if (fragT < 0.6) {
        // Middle: Mix Yellow -> Red
        finalColor = mix(colorYellow, colorRed, (fragT - 0.3) / 0.3);
    } else {
        // Top: Mix Red -> Grey Smoke
        finalColor = mix(colorRed, colorGrey, (fragT - 0.6) / 0.4);
    }

    // 4. Alpha/Transparency
    // Fade out at the very top (lifecycle) AND at the edges of the circle (dist)
    float alpha = 1.0 - fragT; // Fade vertically
    alpha = alpha * (1.0 - dist * dist); // Fade radially (squared for softer edge)

    outColor = vec4(finalColor, alpha);
}