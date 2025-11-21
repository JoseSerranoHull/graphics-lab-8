#version 450

// ---------------------- UBO ----------------------
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

// ---------------------- Inputs (GeometryVertex struct) ----------------------
// Ensure these locations match your C++ getAttributeDescriptions:
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in vec3 inTangent;
layout(location = 5) in vec3 inBinormal;

// ---------------------- Outputs (To Fragment Shader) ----------------------
layout(location = 0) out vec3 fragColor;
layout(location = 3) out vec2 fragTexCoord;

// TBN Vectors and World Position
layout(location = 7) out vec3 fragT; // Tangent (world space)
layout(location = 8) out vec3 fragB; // Binormal (world space)
layout(location = 9) out vec3 fragN; // Normal (world space)
layout(location = 10) out vec3 fragWorldPos; // World Position
layout(location = 11) out vec3 viewDir_tangent; // View direction in tangent space

void main() {
    // Standard MVP transform
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);

    // World-space position
    fragWorldPos = (ubo.model * vec4(inPosition, 1.0)).xyz;

    // Pass color and texcoord
    fragColor = inColor;
    fragTexCoord = inTexCoord;

    // TBN matrix calculation: Transform T, B, N from model-space to world-space
    mat3 normalMatrix = mat3(transpose(inverse(ubo.model)));
    fragT = normalize(normalMatrix * inTangent);
    fragB = normalize(normalMatrix * inBinormal);
    fragN = normalize(normalMatrix * inNormal);

    // Calculate the World Space View Direction (from fragment to eye)
    vec3 V_world = normalize(ubo.eyePos - fragWorldPos);

    // World-to-Tangent-Space matrix is constructed from the TBN vectors:
    mat3 TBN = mat3(fragT, fragB, fragN);

    // Transform World View Direction into Tangent Space
    viewDir_tangent = TBN * V_world;
}