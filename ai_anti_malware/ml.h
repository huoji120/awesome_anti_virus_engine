#pragma once
#include "head.h"
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <unordered_map>

// 前向声明
struct PeInfo;
struct SectionInfo;
class BasicPeInfo;
struct RichEntry {
    uint16_t productId;  // 组件ID
    uint16_t buildId;    // 版本号
    uint32_t useCount;   // 使用次数
};

struct RichHeaderInfo {
    uint32_t checksum;               // 校验和
    std::vector<RichEntry> entries;  // Rich头条目
};
// RVA转换为内存中的指针的辅助函数
inline BYTE* RvaToPtr(DWORD rva, BYTE* peBuffer) {
    if (!peBuffer || rva == 0) return nullptr;

    PIMAGE_NT_HEADERS ntHeaders =
        (PIMAGE_NT_HEADERS)peconv::get_nt_hdrs(peBuffer);
    if (!ntHeaders) return nullptr;

    PIMAGE_SECTION_HEADER section = IMAGE_FIRST_SECTION(ntHeaders);
    WORD numSections = ntHeaders->FileHeader.NumberOfSections;

    for (WORD i = 0; i < numSections; i++, section++) {
        // 检查RVA是否在这个节区范围内
        if (rva >= section->VirtualAddress &&
            rva < section->VirtualAddress + section->Misc.VirtualSize) {
            // 计算文件偏移
            DWORD offset =
                rva - section->VirtualAddress + section->PointerToRawData;
            return peBuffer + offset;
        }
    }

    // 如果RVA在PE头部内
    DWORD sizeOfHeaders = 0;
    bool isX64 = peconv::is64bit(peBuffer);

    if (isX64) {
        PIMAGE_NT_HEADERS64 ntHeaders64 = (PIMAGE_NT_HEADERS64)ntHeaders;
        sizeOfHeaders = ntHeaders64->OptionalHeader.SizeOfHeaders;
    } else {
        PIMAGE_NT_HEADERS32 ntHeaders32 = (PIMAGE_NT_HEADERS32)ntHeaders;
        sizeOfHeaders = ntHeaders32->OptionalHeader.SizeOfHeaders;
    }

    if (rva < sizeOfHeaders) {
        return peBuffer + rva;
    }

    return nullptr;
}

// 在头文件中声明score函数（从外部导入）
extern double score(double* input);

class MachineLearning {
   public:
    MachineLearning();
    ~MachineLearning();
    bool ParseRichHeader(const uint8_t* peBuffer, RichHeaderInfo& richInfo);
    // 提取特征并返回特征向量
    std::vector<double> ExtractFeatures(const uint8_t* buffer,
                                        size_t bufferSize);

    // 将特征导出到CSV
    bool ExportToCSV(const std::vector<double>& features,
                     const std::string& outputPath);

    // 批量处理目录中的样本并生成CSV
    bool ProcessDirectory(const std::string& directoryPath,
                          const std::string& outputCsvPath);

    // 读取文件到内存
    std::vector<uint8_t> ReadFileToBuffer(const std::string& filePath);

    // 新增方法：使用XGBoost模型预测文件是否为恶意软件
    double PredictMalware(const uint8_t* buffer, size_t bufferSize);
    double PredictMalwareFromFile(const std::string& filePath);

   private:
    // 特征提取辅助函数
    std::vector<double> EncodeProperties(
        const PeInfo& peInfo, const std::vector<std::string>& dllTables);
    std::vector<double> EncodeEntrypoint(const std::vector<uint8_t>& epBytes);
    std::vector<double> EncodeHistogram(const uint8_t* data, size_t size);
    std::vector<double> EncodeLibraries(
        const std::vector<std::string>& dllTable);
    std::vector<double> EncodeSections(const std::vector<SectionInfo>& sections,
                                       bool isX64);
    std::tuple<std::vector<double>, std::vector<int>> GetOpcodeStatistics(
        const uint8_t* data, size_t dataSize, bool isX64, const PeInfo& peInfo);
    int GetOpcodeType(const void* code, bool isX64);
    double CalculateEntropy(const uint8_t* data, size_t size);

    // 常量定义
    std::vector<std::string> _properties;
    std::vector<std::string> _libraries;
    std::unordered_map<std::string, int> _opcodeTypeDict;
};

// PE文件信息结构
struct PeInfo {
    uint32_t addressOfEntryPoint;
    uint32_t baseOfCode;
    uint32_t sizeOfCode;
    uint32_t sizeOfImage;
    uint32_t sizeOfHeaders;
    uint32_t characteristics;
    uint32_t dllCharacteristics;
    bool isX64;

    // PE目录标志
    bool hasConfiguration;
    bool hasDebug;
    bool hasExceptions;
    bool hasExports;
    bool hasImports;
    bool hasNx;  // NX兼容标志
    bool hasRelocations;
    bool hasResources;
    bool hasSignatures;
    bool hasTls;
    bool hasDelayImports;
    bool hasImageBase;
    bool hasEntryIat;
    bool hasRich;
};

// 节区信息结构
struct SectionInfo {
    uint32_t characteristics;
    double entropy;
    uint32_t sizeOfRawData;
    uint32_t virtualSize;
};