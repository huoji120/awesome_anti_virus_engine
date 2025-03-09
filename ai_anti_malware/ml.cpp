#include "ml.h"
#include <Windows.h>
#include <array>
#include <limits>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <sstream>
#include <cfloat>
#include <filesystem>
// 确保std命名空间中的函数可用
using std::max;
using std::min;

MachineLearning::MachineLearning() {
    // 初始化属性列表
    _properties = {"has_configuration", "has_debug",     "has_exceptions",
                   "has_exports",       "has_imports",   "has_nx",
                   "has_relocations",   "has_resources", "has_signatures",
                   "has_tls",           "has_entry_iat", "has_image_base",
                   "has_delay_imports", "has_rich"};

    // 初始化库列表
    _libraries = {"libssp-0",
                  "kernel32",
                  "user32",
                  "advapi32",
                  "oleaut32",
                  "shell32",
                  "ole32",
                  "gdi32",
                  "comctl32",
                  "version",
                  "msvcrt",
                  "comdlg32",
                  "shlwapi",
                  "wininet",
                  "ws2_32",
                  "winmm",
                  "winspool.drv",
                  "wsock32",
                  "msvbvm60",
                  "rpcrt4",
                  "mpr",
                  "psapi",
                  "iphlpapi",
                  "ntdll",
                  "msimg32",
                  "mscoree",
                  "crypt32",
                  "gdiplus",
                  "userenv",
                  "crtdll",
                  "oledlg",
                  "mfc42",
                  "urlmon",
                  "imm32",
                  "rtl100.bpl",
                  "netapi32",
                  "wintrust",
                  "vcl100.bpl",
                  "vcl50.bpl",
                  "uxtheme",
                  "setupapi",
                  "ntoskrnl.pe",
                  "msi",
                  "msvcp60",
                  "lz32",
                  "winhttp",
                  "hal",
                  "core.bpl",
                  "rbrcl1416.bpl",
                  "dbghelp",
                  "api-ms-win-crt-runtime-l1-1-0",
                  "api-ms-win-crt-heap-l1-1-0",
                  "api-ms-win-crt-math-l1-1-0",
                  "api-ms-win-crt-stdio-l1-1-0",
                  "api-ms-win-crt-locale-l1-1-0",
                  "oleacc",
                  "komponentyd17.bpl",
                  "job.bpl",
                  "cam.bpl",
                  "vcruntime140",
                  "secur32",
                  "msvcr100",
                  "cxeditorsrs17.bpl",
                  "rasapi32",
                  "api-ms-win-crt-string-l1-1-0",
                  "wtsapi32",
                  "imagehlp",
                  "msvcp140",
                  "cnc.bpl",
                  "indyprotocols190.bpl",
                  "api-ms-win-crt-convert-l1-1-0",
                  "msvcr120",
                  "vcl60.bpl",
                  "rbrcl210.bpl",
                  "rtl170.bpl",
                  "rbide1416.bpl",
                  "rtl60.bpl",
                  "vcl170.bpl",
                  "wldap32",
                  "shfolder",
                  "cxlibraryrs17.bpl",
                  "msvcirt",
                  "report.bpl",
                  "rtl190.bpl",
                  "msvcr90",
                  "api-ms-win-crt-filesystem-l1-1-0",
                  "cxeditorsrs16.bpl",
                  "avifil32",
                  "api-ms-win-crt-time-l1-1-0",
                  "jli",
                  "graphic.bpl",
                  "olepro32",
                  "rtl160.bpl",
                  "spmmachine.bpl",
                  "cabinet",
                  "indycore190.bpl",
                  "sacom210.bpl",
                  "rbrtl1416.bpl",
                  "api-ms-win-crt-utility-l1-1-0",
                  "vcl160.bpl",
                  "api-ms-win-crt-environment-l1-1-0",
                  "zcomponent170.bpl",
                  "msvfw32",
                  "libadm_coreutils6",
                  "rbsha",
                  "dxpscorers16.bpl",
                  "msacm32",
                  "vcl70.bpl",
                  "applicationmanagement.bpl",
                  "jobgui.bpl",
                  "indyprotocols170.bpl",
                  "rtl70.bpl",
                  "cxed210.bpl",
                  "msvcr80",
                  "libadm_coretinypy6",
                  "ucrtbased",
                  "vcruntime140d",
                  "msvcp120",
                  "msvcp140d",
                  "dinput8",
                  "gui.bpl",
                  "maincontrols.bpl",
                  "rtl120.bpl",
                  "jcl170.bpl",
                  "frx17.bpl",
                  "fs17.bpl",
                  "vcl190.bpl",
                  "sdl2",
                  "machine.bpl",
                  "mfc42u",
                  "normaliz",
                  "sdl2_gfx",
                  "sdl2_ttf",
                  "sdl2_mixer",
                  "msvcp80",
                  "cxgridrs17.bpl",
                  "cxeditorsvcld7.bpl",
                  "libeay32",
                  "cxlibraryd11.bpl",
                  "vcl120.bpl",
                  "gr32_d6.bpl",
                  "cxlibraryrs16.bpl",
                  "cxgridrs16.bpl",
                  "vcl40.bpl",
                  "opengl32",
                  "qt5core",
                  "qtcore4",
                  "wdfldr.sys",
                  "nesting.bpl",
                  "fltmgr.sys"};
}

MachineLearning::~MachineLearning() {
    // 析构函数，清理资源（如有必要）
}

bool MachineLearning::ParseRichHeader(const uint8_t* peBuffer,
                                      RichHeaderInfo& richInfo) {
    PIMAGE_DOS_HEADER dosHeader = (PIMAGE_DOS_HEADER)(peBuffer);

    // 检查DOS头部有效性
    if (!dosHeader || dosHeader->e_magic != IMAGE_DOS_SIGNATURE) {
        return false;
    }

    // 搜索范围是DOS头后到PE头前
    const uint32_t* scanPtr =
        reinterpret_cast<const uint32_t*>(peBuffer + sizeof(IMAGE_DOS_HEADER));
    size_t maxItems =
        (dosHeader->e_lfanew - sizeof(IMAGE_DOS_HEADER)) / sizeof(uint32_t);

    // 查找DanS标记
    size_t dansIndex = 0;
    for (; dansIndex < maxItems - 1; dansIndex++) {
        if (scanPtr[dansIndex] == 0x536E6144) {  // "DanS"
            break;
        }
    }

    if (dansIndex >= maxItems - 1) {
        return false;  // 没找到DanS
    }

    // 获取校验和
    uint32_t checksum = scanPtr[dansIndex + 1];
    richInfo.checksum = checksum;

    // 找Rich标记
    size_t richIndex = 0;
    for (richIndex = dansIndex + 2; richIndex < maxItems; richIndex++) {
        if ((scanPtr[richIndex] ^ checksum) ==
            0x68636952) {  // "Rich" ^ checksum
            break;
        }
    }

    if (richIndex >= maxItems) {
        return false;  // 没找到Rich
    }

    // 解析Rich条目
    // DanS之前的数据是Rich条目，每个条目占用2个DWORD
    size_t entryCount = (richIndex - dansIndex - 2) / 2;
    richInfo.entries.reserve(entryCount);

    for (size_t i = 0; i < entryCount; i++) {
        size_t entryPos = richIndex - 2 * (i + 1);
        uint32_t dword1 = scanPtr[entryPos] ^ checksum;
        uint32_t dword2 = scanPtr[entryPos + 1] ^ checksum;

        RichEntry entry;
        entry.productId = dword1 & 0xFFFF;        // 低16位是ProductId
        entry.buildId = (dword1 >> 16) & 0xFFFF;  // 高16位是BuildId
        entry.useCount = dword2;                  // 使用次数

        richInfo.entries.push_back(entry);
    }

    return true;
}

// 添加一个C风格的函数处理SEH部分
auto processImportWithSEH_Internal(const uint8_t* buffer, size_t bufferSize,
                                   char** libNames, size_t* libCount,
                                   size_t maxLibs) -> BOOL {
    __try {
        // 懒得JB处理了,累了.这里是不安全的
        size_t impRva = 0;
        size_t count = 0;
        IMAGE_DATA_DIRECTORY* impDir =
            peconv::get_directory_entry(buffer, IMAGE_DIRECTORY_ENTRY_IMPORT);
        if (impDir) {
            impRva = impDir->VirtualAddress;
            IMAGE_IMPORT_DESCRIPTOR* impDesc =
                reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(
                    RvaToPtr(impRva, (BYTE*)buffer));
            while (impDesc && impDesc->Name != 0 && count < maxLibs) {
                char* libName = reinterpret_cast<char*>(
                    RvaToPtr(impDesc->Name, (BYTE*)buffer));
                if (libName) {
                    libNames[count] = libName;
                    count++;
                }
                impDesc++;
            }
            *libCount = count;
            return TRUE;
        }
        return FALSE;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        printf("skip file:  (access violation)\n");
        return FALSE;
    }
}

auto processImportWithSEH(const uint8_t* buffer, size_t bufferSize,
                          std::vector<std::string>& importedLibraries) -> void {
    const size_t MAX_LIBS = 1000;  // 设置一个合理的最大值
    char* libNames[MAX_LIBS] = {0};
    size_t libCount = 0;

    // 调用处理SEH的内部函数
    if (processImportWithSEH_Internal(buffer, bufferSize, libNames, &libCount,
                                      MAX_LIBS)) {
        // 将结果转换为C++对象
        for (size_t i = 0; i < libCount; i++) {
            if (libNames[i]) {
                std::string libNameStr = libNames[i];
                std::transform(libNameStr.begin(), libNameStr.end(),
                               libNameStr.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                importedLibraries.push_back(libNameStr);
            }
        }
    }
}

std::vector<double> MachineLearning::ExtractFeatures(const uint8_t* buffer,
                                                     size_t bufferSize) {
    // 使用libpeconv解析PE文件
    size_t v_size = 0;
    BYTE* peBuffer = peconv::load_pe_module(const_cast<BYTE*>(buffer),
                                            bufferSize, v_size, false, false);
    if (!peBuffer) {
        return std::vector<double>();
    }

    // 解析PE信息
    PeInfo peInfo;
    std::vector<SectionInfo> sections;
    std::vector<std::string> importedLibraries;
    std::vector<uint8_t> entrypointBytes;

    // 检查是否为64位PE
    peInfo.isX64 = peconv::is64bit(peBuffer);

    // 获取PE头信息
    PIMAGE_NT_HEADERS ntHeaders =
        (PIMAGE_NT_HEADERS)peconv::get_nt_hdrs(peBuffer);
    if (!ntHeaders) {
        peconv::free_pe_buffer(peBuffer);
        return std::vector<double>();
    }

    // 从NT头部获取信息
    if (peInfo.isX64) {
        // 64位PE文件
        PIMAGE_NT_HEADERS64 ntHeaders64 = (PIMAGE_NT_HEADERS64)ntHeaders;
        peInfo.addressOfEntryPoint =
            ntHeaders64->OptionalHeader.AddressOfEntryPoint;
        peInfo.baseOfCode = ntHeaders64->OptionalHeader.BaseOfCode;
        peInfo.sizeOfCode = ntHeaders64->OptionalHeader.SizeOfCode;
        peInfo.sizeOfImage = ntHeaders64->OptionalHeader.SizeOfImage;
        peInfo.sizeOfHeaders = ntHeaders64->OptionalHeader.SizeOfHeaders;
        peInfo.characteristics = ntHeaders64->FileHeader.Characteristics;
        peInfo.dllCharacteristics =
            ntHeaders64->OptionalHeader.DllCharacteristics;
        peInfo.hasImageBase = ntHeaders64->OptionalHeader.ImageBase != 0;
    } else {
        // 32位PE文件
        PIMAGE_NT_HEADERS32 ntHeaders32 = (PIMAGE_NT_HEADERS32)ntHeaders;
        peInfo.addressOfEntryPoint =
            ntHeaders32->OptionalHeader.AddressOfEntryPoint;
        peInfo.baseOfCode = ntHeaders32->OptionalHeader.BaseOfCode;
        peInfo.sizeOfCode = ntHeaders32->OptionalHeader.SizeOfCode;
        peInfo.sizeOfImage = ntHeaders32->OptionalHeader.SizeOfImage;
        peInfo.sizeOfHeaders = ntHeaders32->OptionalHeader.SizeOfHeaders;
        peInfo.characteristics = ntHeaders32->FileHeader.Characteristics;
        peInfo.dllCharacteristics =
            ntHeaders32->OptionalHeader.DllCharacteristics;
        peInfo.hasImageBase = ntHeaders32->OptionalHeader.ImageBase != 0;
    }

    // 检查PE目录
    IMAGE_DATA_DIRECTORY* dataDir = peconv::get_directory_entry(
        peBuffer, IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR);
    peInfo.hasConfiguration = dataDir && dataDir->VirtualAddress != 0;

    dataDir =
        peconv::get_directory_entry(peBuffer, IMAGE_DIRECTORY_ENTRY_DEBUG);
    peInfo.hasDebug = dataDir && dataDir->VirtualAddress != 0;

    dataDir =
        peconv::get_directory_entry(peBuffer, IMAGE_DIRECTORY_ENTRY_EXCEPTION);
    peInfo.hasExceptions = dataDir && dataDir->VirtualAddress != 0;

    dataDir =
        peconv::get_directory_entry(peBuffer, IMAGE_DIRECTORY_ENTRY_EXPORT);
    peInfo.hasExports = dataDir && dataDir->VirtualAddress != 0;

    dataDir =
        peconv::get_directory_entry(peBuffer, IMAGE_DIRECTORY_ENTRY_IMPORT);
    peInfo.hasImports = dataDir && dataDir->VirtualAddress != 0;

    // NX标志检查
    peInfo.hasNx =
        (peInfo.dllCharacteristics & IMAGE_DLLCHARACTERISTICS_NX_COMPAT) != 0;

    dataDir =
        peconv::get_directory_entry(peBuffer, IMAGE_DIRECTORY_ENTRY_BASERELOC);
    peInfo.hasRelocations = dataDir && dataDir->VirtualAddress != 0;

    dataDir =
        peconv::get_directory_entry(peBuffer, IMAGE_DIRECTORY_ENTRY_RESOURCE);
    peInfo.hasResources = dataDir && dataDir->VirtualAddress != 0;

    dataDir =
        peconv::get_directory_entry(peBuffer, IMAGE_DIRECTORY_ENTRY_SECURITY);
    peInfo.hasSignatures = dataDir && dataDir->VirtualAddress != 0;

    dataDir = peconv::get_directory_entry(peBuffer, IMAGE_DIRECTORY_ENTRY_TLS);
    peInfo.hasTls = dataDir && dataDir->VirtualAddress != 0;

    dataDir = peconv::get_directory_entry(peBuffer,
                                          IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT);
    peInfo.hasDelayImports = dataDir && dataDir->VirtualAddress != 0;

    dataDir = peconv::get_directory_entry(peBuffer, IMAGE_DIRECTORY_ENTRY_IAT);
    peInfo.hasEntryIat = dataDir && dataDir->VirtualAddress != 0;

    // Rich头部检测 - 安全实现
    peInfo.hasRich = false;
    PIMAGE_DOS_HEADER dosHeader = reinterpret_cast<PIMAGE_DOS_HEADER>(peBuffer);
    if (dosHeader && dosHeader->e_magic == IMAGE_DOS_SIGNATURE) {
        // 确保e_lfanew值合理
        if (dosHeader->e_lfanew > sizeof(IMAGE_DOS_HEADER) &&
            dosHeader->e_lfanew < v_size) {  // 确保在PE文件大小范围内

            size_t maxLen = dosHeader->e_lfanew - sizeof(IMAGE_DOS_HEADER);
            // 确保搜索区域不会太大（预防恶意构造的文件）
            const size_t MAX_RICH_SEARCH_SIZE = 1024;  // 合理的Rich头最大区域
            if (maxLen > MAX_RICH_SEARCH_SIZE) {
                maxLen = MAX_RICH_SEARCH_SIZE;
            }

            // 确保不会越界
            if (sizeof(IMAGE_DOS_HEADER) + maxLen <= v_size) {
                const uint32_t* richPtr = reinterpret_cast<const uint32_t*>(
                    peBuffer + sizeof(IMAGE_DOS_HEADER));

                // 确保剩余长度至少能容纳一个uint32_t
                for (size_t i = 0;
                     i < maxLen / 4 - 1 && (i + 1) * sizeof(uint32_t) <= maxLen;
                     i++) {
                    if (richPtr[i] == 0x68636952) {  // "Rich"
                        peInfo.hasRich = true;
                        break;
                    }
                }
            }
        }
    }

    // 获取导入DLL列表
    if (peInfo.hasImports) {
        processImportWithSEH(peBuffer, bufferSize, importedLibraries);
    }

    // 获取节区信息
    size_t sectionsCount = peconv::get_sections_count(peBuffer, bufferSize);
    for (size_t i = 0; i < sectionsCount; i++) {
        PIMAGE_SECTION_HEADER section =
            peconv::get_section_hdr(peBuffer, bufferSize, i);
        if (!section) continue;

        SectionInfo secInfo;
        secInfo.characteristics = section->Characteristics;
        secInfo.sizeOfRawData = section->SizeOfRawData;
        secInfo.virtualSize = section->Misc.VirtualSize;

        // 计算节区熵
        BYTE* sectionData = RvaToPtr(section->VirtualAddress, peBuffer);
        secInfo.entropy =
            (sectionData && section->SizeOfRawData > 0)
                ? CalculateEntropy(sectionData, section->SizeOfRawData)
                : 0.0;

        sections.push_back(secInfo);
    }

    // 获取入口点前255字节
    if (peInfo.addressOfEntryPoint > 0) {
        BYTE* epPtr = RvaToPtr(peInfo.addressOfEntryPoint, peBuffer);
        if (epPtr) {
            // 确保不会越界
            size_t maxBytes =
                std::min<size_t>(255, bufferSize - (epPtr - peBuffer));
            entrypointBytes.assign(epPtr, epPtr + maxBytes);
        }
    }

    // 提取所有特征
    std::vector<double> allFeatures;

    // 1. PE段属性
    std::vector<double> propFeatures =
        EncodeProperties(peInfo, importedLibraries);
    allFeatures.insert(allFeatures.end(), propFeatures.begin(),
                       propFeatures.end());

    // 2. 导入DLL检测
    std::vector<double> libFeatures = EncodeLibraries(importedLibraries);
    allFeatures.insert(allFeatures.end(), libFeatures.begin(),
                       libFeatures.end());

    // 3. 文件熵
    double fileEntropy = CalculateEntropy(buffer, bufferSize);
    allFeatures.push_back(fileEntropy);

    // 4. 入口点前255字节
    std::vector<double> epFeatures = EncodeEntrypoint(entrypointBytes);
    allFeatures.insert(allFeatures.end(), epFeatures.begin(), epFeatures.end());

    // 5. 节区信息
    std::vector<double> secFeatures = EncodeSections(sections, peInfo.isX64);
    allFeatures.insert(allFeatures.end(), secFeatures.begin(),
                       secFeatures.end());

    // 6. 文件和代码段的比率
    double codeRatio =
        (peInfo.sizeOfCode > 0 && peInfo.sizeOfImage > 0)
            ? static_cast<double>(peInfo.sizeOfCode) / peInfo.sizeOfImage
            : 0.0;
    allFeatures.push_back(codeRatio);

    // 7. 节区数量
    allFeatures.push_back(static_cast<double>(sections.size()));

    // 清理资源
    peconv::free_pe_buffer(peBuffer);

    return allFeatures;
}

std::vector<double> MachineLearning::EncodeProperties(
    const PeInfo& peInfo, const std::vector<std::string>& dllTables) {
    std::vector<double> features;

    // 添加各属性的布尔值（转为double: 1.0=true, 0.0=false）
    features.push_back(peInfo.hasConfiguration ? 1.0 : 0.0);
    features.push_back(peInfo.hasDebug ? 1.0 : 0.0);
    features.push_back(peInfo.hasExceptions ? 1.0 : 0.0);
    features.push_back(peInfo.hasExports ? 1.0 : 0.0);
    features.push_back(peInfo.hasImports ? 1.0 : 0.0);
    features.push_back(peInfo.hasNx ? 1.0 : 0.0);
    features.push_back(peInfo.hasRelocations ? 1.0 : 0.0);
    features.push_back(peInfo.hasResources ? 1.0 : 0.0);
    features.push_back(peInfo.hasSignatures ? 1.0 : 0.0);
    features.push_back(peInfo.hasTls ? 1.0 : 0.0);
    features.push_back(peInfo.hasEntryIat ? 1.0 : 0.0);
    features.push_back(peInfo.hasImageBase ? 1.0 : 0.0);
    features.push_back(peInfo.hasDelayImports ? 1.0 : 0.0);
    features.push_back(peInfo.hasRich ? 1.0 : 0.0);

    return features;
}

std::vector<double> MachineLearning::EncodeEntrypoint(
    const std::vector<uint8_t>& epBytes) {
    std::vector<double> features;

    // 只使用前64个字节，确保特征数量固定
    size_t bytesToUse = std::min<size_t>(64, epBytes.size());

    // 原始字节转为浮点值（按Python代码中的normalize处理）
    for (size_t i = 0; i < bytesToUse; i++) {
        features.push_back(static_cast<double>(epBytes[i]) / 255.0);
    }

    // 填充至64字节长度
    while (features.size() < 64) {
        features.push_back(0.0);
    }

    return features;
}

std::vector<double> MachineLearning::EncodeHistogram(const uint8_t* data,
                                                     size_t size) {
    std::vector<double> features(256, 0.0);

    if (data && size > 0) {
        // 统计字节频率
        for (size_t i = 0; i < size; i++) {
            features[data[i]]++;
        }

        // 归一化频率
        for (auto& freq : features) {
            freq /= static_cast<double>(size);
        }
    }

    return features;
}

std::vector<double> MachineLearning::EncodeLibraries(
    const std::vector<std::string>& importedLibraries) {
    std::vector<double> features(_libraries.size(), 0.0);

    // 检查每个库是否被导入
    for (size_t i = 0; i < _libraries.size(); i++) {
        const std::string& lib = _libraries[i];
        for (const auto& imported : importedLibraries) {
            if (imported.find(lib) != std::string::npos) {
                features[i] = 1.0;
                break;
            }
        }
    }

    return features;
}

std::vector<double> MachineLearning::EncodeSections(
    const std::vector<SectionInfo>& sections, bool isX64) {
    std::vector<double> features;
    size_t numSections = sections.size();
    if (numSections == 0) {
        return std::vector<double>(5, 0.0);  // 返回全零特征
    }

    // 计算熵特征
    double totalEntropy = 0.0;
    double maxEntropy = 0.0;
    for (const auto& sec : sections) {
        totalEntropy += sec.entropy;
        if (sec.entropy > maxEntropy) {
            maxEntropy = sec.entropy;
        }
    }
    double avgEntropy = totalEntropy / numSections;
    double normAvgEntropy = (maxEntropy > 0) ? avgEntropy / maxEntropy : 0.0;

    // 计算大小比率
    double maxSize = 0.0;
    double minVSize = DBL_MAX;
    for (const auto& sec : sections) {
        if (static_cast<double>(sec.sizeOfRawData) > maxSize) {
            maxSize = static_cast<double>(sec.sizeOfRawData);
        }
        if (sec.virtualSize > 0 &&
            static_cast<double>(sec.virtualSize) < minVSize) {
            minVSize = static_cast<double>(sec.virtualSize);
        }
    }

    // 根据PE文件类型调整计算方式
    double normSize = 0.0;
    if (minVSize > 0 && minVSize != DBL_MAX) {
        if (isX64) {
            // 64位PE文件可能有更大的对齐要求
            normSize = maxSize / (minVSize * 2.0);
        } else {
            // 32位PE文件的处理方式
            normSize = maxSize / minVSize;
        }
    }

    // 返回特征
    features.push_back(static_cast<double>(numSections));
    features.push_back(avgEntropy);
    features.push_back(maxEntropy);
    features.push_back(normAvgEntropy);
    features.push_back(normSize);

    return features;
}

double MachineLearning::CalculateEntropy(const uint8_t* data, size_t size) {
    // 基本参数检查
    if (!data || size == 0) {
        return 0.0;
    }

    // 添加合理性检查，防止过大的size造成计算问题或DoS攻击
    // 通常PE文件不应超过一定大小，这里设置上限为2GB
    constexpr size_t MAX_SAFE_SIZE = 2ULL * 1024 * 1024 * 1024;  // 2GB
    if (size > MAX_SAFE_SIZE) {
        return 0.0;
    }

    std::array<double, 256> frequencies = {};
    __try {
        // 懒得JB处理了,累了.这里是不安全的
        //  统计每个字节的频率
        for (size_t i = 0; i < size; i++) {
            uint8_t byteValue = data[i];
            frequencies[byteValue] += 1.0;
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        printf("skip file:  (access violation)\n");
    }

    // 计算香农熵
    double entropy = 0.0;
    for (const auto& freq : frequencies) {
        if (freq > 0) {
            double p = freq / static_cast<double>(size);
            entropy -= p * std::log2(p);
        }
    }

    return entropy;
}

bool MachineLearning::ExportToCSV(const std::vector<double>& features,
                                  const std::string& outputPath) {
    std::ofstream outFile(outputPath);
    if (!outFile.is_open()) {
        std::cerr << "无法打开输出文件: " << outputPath << std::endl;
        return false;
    }

    // 写入特征
    for (size_t i = 0; i < features.size(); i++) {
        outFile << std::fixed << std::setprecision(6) << features[i];
        if (i < features.size() - 1) {
            outFile << ",";
        }
    }
    outFile << std::endl;

    outFile.close();
    return true;
}

int MachineLearning::GetOpcodeType(const void* code, bool isX64) {
    // 此函数未使用，但保留实现接口
    return 0;
}

std::tuple<std::vector<double>, std::vector<int>>
MachineLearning::GetOpcodeStatistics(const uint8_t* data, size_t dataSize,
                                     bool isX64, const PeInfo& peInfo) {
    // 此函数未使用，但保留实现接口
    return std::make_tuple(std::vector<double>(), std::vector<int>());
}

std::vector<uint8_t> MachineLearning::ReadFileToBuffer(
    const std::string& filePath) {
    std::ifstream fileStream(filePath, std::ios::binary | std::ios::ate);
    if (!fileStream.is_open()) {
        std::cerr << "无法打开文件: " << filePath << std::endl;
        return std::vector<uint8_t>();
    }

    // 获取文件大小
    std::streamsize fileSize = fileStream.tellg();
    fileStream.seekg(0, std::ios::beg);

    // 分配缓冲区并读取文件
    std::vector<uint8_t> buffer(fileSize);
    if (!fileStream.read(reinterpret_cast<char*>(buffer.data()), fileSize)) {
        std::cerr << "读取文件失败: " << filePath << std::endl;
        return std::vector<uint8_t>();
    }

    return buffer;
}

bool MachineLearning::ProcessDirectory(const std::string& directoryPath,
                                       const std::string& outputCsvPath) {
    // 检查文件是否已存在
    bool fileExists = std::filesystem::exists(outputCsvPath);

    // 打开CSV文件用于写入，如果文件已存在则使用追加模式
    std::ofstream csvFile;
    if (fileExists) {
        csvFile.open(outputCsvPath, std::ios::app);
    } else {
        csvFile.open(outputCsvPath);
    }

    if (!csvFile.is_open()) {
        std::cerr << "无法创建或打开CSV文件: " << outputCsvPath << std::endl;
        return false;
    }

    // 仅在文件不存在时写入CSV标题行
    /*
   if (!fileExists) {

       // 写入CSV标题行
       csvFile << "文件路径";
       for (size_t i = 0; i < _properties.size(); i++) {
           csvFile << ",属性_" << i;
       }
       for (size_t i = 0; i < _libraries.size(); i++) {
           csvFile << ",库_" << i;
       }
       csvFile << ",文件熵";
       for (size_t i = 0; i < 64; i++) {  // 前64个字节特征
           csvFile << ",EP_" << i;
       }
       csvFile << ",节区数";
       csvFile << ",平均熵";
       csvFile << ",最大熵";
       csvFile << ",归一化平均熵";
       csvFile << ",节区大小比率";
       csvFile << ",代码比率";
       csvFile << ",节区计数";
       csvFile << std::endl;

    }
 */
    // 递归遍历目录
    WIN32_FIND_DATAA findData;
    std::string searchPath = directoryPath + "\\*";
    HANDLE hFind = FindFirstFileA(searchPath.c_str(), &findData);

    if (hFind == INVALID_HANDLE_VALUE) {
        std::cerr << "无法访问目录: " << directoryPath << std::endl;
        csvFile.close();
        return false;
    }

    int processedCount = 0;
    int failedCount = 0;

    do {
        // 跳过 "." 和 ".." 目录
        if (strcmp(findData.cFileName, ".") == 0 ||
            strcmp(findData.cFileName, "..") == 0) {
            continue;
        }

        std::string currentPath = directoryPath + "\\" + findData.cFileName;

        if (findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            // 递归处理子目录
            ProcessDirectory(currentPath, outputCsvPath);
        } else {
            // 处理文件
            std::vector<uint8_t> fileBuffer = ReadFileToBuffer(currentPath);
            if (fileBuffer.empty()) {
                std::cerr << "skip file: " << currentPath << " (read failed)"
                          << std::endl;
                failedCount++;
                continue;
            }

            // 提取特征
            std::vector<double> features =
                ExtractFeatures(fileBuffer.data(), fileBuffer.size());
            if (features.empty()) {
                std::cerr << "skip file: " << currentPath
                          << " (can't get feature)" << std::endl;
                failedCount++;
                continue;
            }

            // 写入CSV
            csvFile << currentPath;
            for (const auto& feature : features) {
                csvFile << "," << std::fixed << std::setprecision(6) << feature;
            }
            csvFile << std::endl;

            processedCount++;
            if (processedCount % 100 == 0) {
                std::cout << "a ready processed " << processedCount
                          << " files..." << std::endl;
            }
        }
    } while (FindNextFileA(hFind, &findData));

    FindClose(hFind);
    csvFile.close();
    printf("ML Process Result, success count: %d fail count: %d \n",
           processedCount, failedCount);
    return true;
}

double MachineLearning::PredictMalware(const uint8_t* buffer,
                                       size_t bufferSize) {
    // 提取特征
    std::vector<double> features = ExtractFeatures(buffer, bufferSize);

    // 如果特征提取失败，返回-1.0表示无法预测
    if (features.empty()) {
        return -1.0;
    }

    // 将特征向量传递给XGBoost模型
    return score(features.data());
}
//返回的是白文件的概率
double MachineLearning::PredictMalwareFromFile(const std::string& filePath) {
    // 读取文件
    std::vector<uint8_t> fileBuffer = ReadFileToBuffer(filePath);
    if (fileBuffer.empty()) {
        std::cerr << "无法读取文件: " << filePath << std::endl;
        return -1.0;
    }

    // 使用缓冲区进行预测
    return PredictMalware(fileBuffer.data(), fileBuffer.size());
}