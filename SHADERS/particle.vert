#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 lightPos1;
    vec3 lightColor1;
    vec3 lightPos2;
    vec3 lightColor2;
    vec3 eyePos;
} ubo;

// We will use a Push Constant to pass the Time
layout(push_constant) uniform PushConsts {
    float time;
} pushConsts;

// Input: x,y = quad corners (-1 to 1), z = unique seed (0 to 1)
layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out float fragT; // Lifecycle of particle (0 to 1)

// Settings
float particleSpeed = 0.5;
float particleSpread = 1.5;
float particleSize = 0.5;

// ... (Keep existing variables) ...

void main() {
    // 1. Calculate "t" (Lifecycle)
    float t = fract(inPosition.z + particleSpeed * pushConsts.time);
    fragT = t;

    // 2. Physics
    vec3 centerPos;
    centerPos.y = 2.5 * t; // Move up slightly faster

    // --- FIX IS HERE ---
    // OLD: ... * (1.0 - t)); // Starts wide, ends narrow
    // NEW: ... * t);         // Starts narrow, ends wide (Bonfire shape)
    
    // We add a small constant (0.2) so the base isn't a single infinite point
    float spreadFactor = (t * 0.5 + 0.2); 

    centerPos.x = particleSpread * spreadFactor * (sin(t * 10.0 + inPosition.z * 20.0)); 
    centerPos.z = particleSpread * spreadFactor * (cos(t * 5.0 + inPosition.z * 10.0));
    // -------------------

    // 3. Billboarding (Keep exactly as is)
    vec3 cameraRight = vec3(ubo.view[0][0], ubo.view[1][0], ubo.view[2][0]);
    vec3 cameraUp    = vec3(ubo.view[0][1], ubo.view[1][1], ubo.view[2][1]);

    // Optional: Make particles smaller as they die to simulate burning out
    // float currentSize = particleSize * (1.0 - t * 0.5);
    vec3 offset = (cameraRight * inPosition.x + cameraUp * inPosition.y) * particleSize;

    vec3 finalPos = centerPos + offset;

    gl_Position = ubo.proj * ubo.view * vec4(finalPos, 1.0);
    fragTexCoord = inPosition.xy * 0.5 + 0.5;
}