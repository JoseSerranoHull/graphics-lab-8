#include "MeshLoader.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

GeometryUtils::MeshData MeshLoader::LoadMeshFromFile(const std::string& path)
{
	GeometryUtils::MeshData meshData;

	Assimp::Importer importer;

    // Change the string literal to use double backslashes for Windows file paths
    const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);

	aiMesh* mesh = scene->mMeshes[0];

	for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
		GeometryVertex vertex;
		vertex.pos = {
			mesh->mVertices[i].x,
			mesh->mVertices[i].y,
			mesh->mVertices[i].z
		};
		vertex.color = { 1.0f, 1.0f, 1.0f };
		meshData.vertices.push_back(vertex);
	}

	for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
		aiFace face = mesh->mFaces[i];
		for (unsigned int j = 0; j < face.mNumIndices; j++) {
			meshData.indices.push_back(face.mIndices[j]);
		}
	}

	return meshData;
}