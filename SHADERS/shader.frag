#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 lightPos1;
    vec3 lightColor1;
    vec3 lightPos2;
    vec3 lightColor2;
    vec3 eyePos;
} ubo;

layout(push_constant) uniform MaterialPushConstant {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
} material;

// Texture samplers
layout(set = 0, binding = 1) uniform sampler2D colSampler;
layout(set = 0, binding = 2) uniform sampler2D normalSampler;
layout(set = 0, binding = 3) uniform sampler2D heightSampler;
// --- NEW SKYBOX SAMPLER ---
layout(set = 0, binding = 4) uniform samplerCube skySampler;

// Inputs
layout(location = 0) in vec3 fragColor;
layout(location = 3) in vec2 fragTexCoord;
layout(location = 7) in vec3 fragT;
layout(location = 8) in vec3 fragB;
layout(location = 9) in vec3 fragN;
layout(location = 10) in vec3 fragWorldPos;
layout(location = 11) in vec3 viewDir_tangent;

layout(location = 0) out vec4 outColor;

void main() {
    // --- 1. NORMAL CALCULATION ---
    mat3 TBN = mat3(fragT, fragB, fragN);
    vec3 N_world;

    // [OPTION A] Use Bump/Normal Mapping (Detailed surface)
    vec3 normalSample = texture(normalSampler, fragTexCoord).rgb;
    vec3 N_tangent = normalize(normalSample * 2.0 - 1.0);
    N_world = normalize(TBN * N_tangent);

    // [OPTION B] Use Flat Geometry Normal (Smooth surface)
    N_world = normalize(fragN); 


    // --- 2. VIEW VECTOR ---
    // Vector from Fragment to Camera (Eye)
    vec3 V = normalize(ubo.eyePos - fragWorldPos); 


    // --- 3. CHOOSE EFFECT ---

    // [EFFECT 1] REFLECTION (Mirror)
    // -------------------------------------------------
    // Incident vector I = -V (Camera to Fragment)
    // vec3 R_reflect = reflect(-V, N_world); 
    // vec3 color = texture(skySampler, R_reflect).rgb;
    // -------------------------------------------------

    // [EFFECT 2] REFRACTION (Glass/Water)
    // -------------------------------------------------
    // Ratio of indices of refraction (Air / Glass)
    float IOR = 1.00 / 1.33; 
    
    // Calculate refraction vector
    // refract(Incident, Normal, eta)
    vec3 R_refract = refract(-V, N_world, IOR);
    
    // Sample the skybox using the refracted vector
    vec3 color = texture(skySampler, R_refract).rgb;
    // -------------------------------------------------


    // --- 4. FINAL OUTPUT ---
    
    // Pure Refraction (Glass look)
    outColor = vec4(color, 1.0);
    
    // Optional: Mix with base color if you want "tinted" glass
    // vec3 albedo = texture(colSampler, fragTexCoord).rgb;
    // outColor = vec4(mix(albedo, color, 0.3), 1.0);
}