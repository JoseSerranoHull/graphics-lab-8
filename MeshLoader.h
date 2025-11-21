#pragma once
#include "GeometryUtils.h"
#include <string>

namespace MeshLoader
{
	GeometryUtils::MeshData LoadMeshFromFile(const std::string& path);
};