#pragma once
#define LOG_LEVEL 1

#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <WinSock2.h>

#include <vector>
#include <map>
#include <unordered_map>
#include <ctime>

#include <functional>
#include <string>
#include <vector>
#include "unicorn/include/unicorn/unicorn.h"
#include "capstone/include/capstone/capstone.h"
#pragma comment(lib, "unicorn/unicorn.lib")
#pragma comment(lib, "capstone/capstone.lib")
#include "libpeconv/include/peconv.h"
#include "native_struct.h"
#include "tiny_wfp_structs.h"
struct BasicPeInfo {
    std::string inputFilePath;
    bool isX64;
    uint64_t RecImageBase;
    uint64_t entryPoint;
    uint64_t imageEnd;
    bool isRelocated;
    uint8_t* peBuffer;
    size_t peSize;
    PIMAGE_NT_HEADERS ntHead64;
    PIMAGE_NT_HEADERS32 ntHead32;
    bool isDll;
};
#include "sandbox.h"
#include "ml.h"