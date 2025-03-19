#include "sandbox.h"
#include "sandbox_callbacks.h"
#include "sandbox_api_winhttp.h"
#include <tlhelp32.h>

// 内部实现函数，处理实际的模块句柄获取逻辑
auto GetModuleHandleInternal(void* sandbox, const std::wstring& moduleName)
    -> HMODULE {
    auto* sb = static_cast<Sandbox*>(sandbox);

    // 如果模块名为空，返回当前进程的基址
    if (moduleName.empty()) {
        return reinterpret_cast<HMODULE>(sb->GetPeInfo()->RecImageBase);
    }

    // 在已加载的模块中查找
    for (const auto& module : sb->GetModuleList()) {
        std::string currentModuleNameA = module->name;
        std::wstring currentModuleName =
            std::wstring(currentModuleNameA.begin(), currentModuleNameA.end());
        if (_wcsicmp(currentModuleName.c_str(), moduleName.c_str()) == 0) {
            return reinterpret_cast<HMODULE>(module->base);
        }
    }

    return nullptr;
}

// GetModuleHandleA的实现
auto Api_GetModuleHandleA(void* sandbox, uc_engine* uc, uint64_t address)
    -> void {
    auto* sb = static_cast<Sandbox*>(sandbox);
    uint64_t esp = 0, rsp = 0;
    HMODULE result = nullptr;

    if (sb->GetPeInfo()->isX64) {
        // 获取第一个参数 (rcx)
        uint64_t moduleNamePtr;
        uc_reg_read(uc, UC_X86_REG_RCX, &moduleNamePtr);
        uc_reg_read(uc, UC_X86_REG_RSP, &rsp);

        std::string moduleName;
        if (moduleNamePtr != 0) {
            // 读取ANSI字符串
            char ch;
            size_t i = 0;
            do {
                if (uc_mem_read(uc, moduleNamePtr + i, &ch, 1) != UC_ERR_OK) {
                    break;
                }
                if (ch == 0) break;
                moduleName += ch;
                i++;
            } while (i < MAX_PATH);
        }

        // 转换为宽字符
        std::wstring wModuleName;
        if (!moduleName.empty()) {
            wModuleName = std::wstring(moduleName.begin(), moduleName.end());
        }

        // 获取模块句柄
        result = GetModuleHandleInternal(sandbox, wModuleName);

        // 设置返回值
        uc_reg_write(uc, UC_X86_REG_RAX, &result);

    } else {
        // 32位实现
        uc_reg_read(uc, UC_X86_REG_ESP, &esp);
        uint32_t moduleNamePtr;
        uc_mem_read(uc, esp + 4, &moduleNamePtr, sizeof(moduleNamePtr));

        std::string moduleName;
        if (moduleNamePtr != 0) {
            // 读取ANSI字符串
            char ch;
            size_t i = 0;
            do {
                if (uc_mem_read(uc, moduleNamePtr + i, &ch, 1) != UC_ERR_OK) {
                    break;
                }
                if (ch == 0) break;
                moduleName += ch;
                i++;
            } while (i < MAX_PATH);
        }

        // 转换为宽字符
        std::wstring wModuleName;
        if (!moduleName.empty()) {
            wModuleName = std::wstring(moduleName.begin(), moduleName.end());
        }

        // 获取模块句柄
        result = GetModuleHandleInternal(sandbox, wModuleName);

        // 设置返回值
        uint32_t result32 = reinterpret_cast<uint32_t>(result);
        uc_reg_write(uc, UC_X86_REG_EAX, &result32);
    }

    // 设置错误码
    DWORD error = result ? 0 : ERROR_MOD_NOT_FOUND;
    if (sb->GetPeInfo()->isX64) {
        sb->GetTeb64()->LastErrorValue = error;
    } else {
        sb->GetTeb32()->LastErrorValue = error;
    }
}

// GetModuleHandleW的实现
auto Api_GetModuleHandleW(void* sandbox, uc_engine* uc, uint64_t address)
    -> void {
    auto* sb = static_cast<Sandbox*>(sandbox);
    uint64_t esp = 0, rsp = 0;
    HMODULE result = nullptr;

    if (sb->GetPeInfo()->isX64) {
        // 获取第一个参数 (rcx)
        uint64_t moduleNamePtr;
        uc_reg_read(uc, UC_X86_REG_RCX, &moduleNamePtr);
        uc_reg_read(uc, UC_X86_REG_RSP, &rsp);

        std::wstring moduleName;
        if (moduleNamePtr != 0) {
            // 读取宽字符串
            wchar_t ch;
            size_t i = 0;
            do {
                if (uc_mem_read(uc, moduleNamePtr + (i * 2), &ch, 2) !=
                    UC_ERR_OK) {
                    break;
                }
                if (ch == 0) break;
                moduleName += ch;
                i++;
            } while (i < MAX_PATH);
        }

        // 获取模块句柄
        result = GetModuleHandleInternal(sandbox, moduleName);

        // 设置返回值
        uc_reg_write(uc, UC_X86_REG_RAX, &result);

    } else {
        // 32位实现
        uc_reg_read(uc, UC_X86_REG_ESP, &esp);
        uint32_t moduleNamePtr;
        uc_mem_read(uc, esp + 4, &moduleNamePtr, sizeof(moduleNamePtr));

        std::wstring moduleName;
        if (moduleNamePtr != 0) {
            // 读取宽字符串
            wchar_t ch;
            size_t i = 0;
            do {
                if (uc_mem_read(uc, moduleNamePtr + (i * 2), &ch, 2) !=
                    UC_ERR_OK) {
                    break;
                }
                if (ch == 0) break;
                moduleName += ch;
                i++;
            } while (i < MAX_PATH);
        }

        // 获取模块句柄
        result = GetModuleHandleInternal(sandbox, moduleName);

        // 设置返回值
        uint32_t result32 = reinterpret_cast<uint32_t>(result);
        uc_reg_write(uc, UC_X86_REG_EAX, &result32);
    }

    // 设置错误码
    DWORD error = result ? 0 : ERROR_MOD_NOT_FOUND;
    if (sb->GetPeInfo()->isX64) {
        sb->GetTeb64()->LastErrorValue = error;
    } else {
        sb->GetTeb32()->LastErrorValue = error;
    }
}
auto Api_VirtualQuery(void* sandbox, uc_engine* uc, uint64_t address) -> void {
    auto* context = static_cast<Sandbox*>(sandbox);
    uint64_t lpAddress = 0;
    uint64_t lpBuffer = 0;
    uint32_t dwLength = 0;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        // 64位参数获取
        uc_reg_read(uc, UC_X86_REG_RCX, &lpAddress);
        uc_reg_read(uc, UC_X86_REG_RDX, &lpBuffer);
        uint64_t temp_length = 0;
        uc_reg_read(uc, UC_X86_REG_R8, &temp_length);
        dwLength = static_cast<uint32_t>(temp_length);
    } else {
        // 32位参数获取
        uint32_t esp = 0;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp);
        esp += 0x4;  // 跳过返回地址

        uint32_t temp_address = 0;
        uint32_t temp_buffer = 0;
        uc_mem_read(uc, esp, &temp_address, sizeof(uint32_t));
        uc_mem_read(uc, esp + 0x4, &temp_buffer, sizeof(uint32_t));
        uc_mem_read(uc, esp + 0x8, &dwLength, sizeof(uint32_t));

        lpAddress = temp_address;
        lpBuffer = temp_buffer;
    }

    // 构造MEMORY_BASIC_INFORMATION结构
    MEMORY_BASIC_INFORMATION mbi = {};
    mbi.BaseAddress =
        reinterpret_cast<void*>(static_cast<uintptr_t>(lpAddress));
    mbi.AllocationBase = mbi.BaseAddress;
    mbi.AllocationProtect = PAGE_EXECUTE_READWRITE;  // 默认保护属性
    mbi.RegionSize = 0x1000;                         // 默认页大小
    mbi.State = MEM_COMMIT;
    mbi.Protect = PAGE_EXECUTE_READWRITE;
    mbi.Type = MEM_PRIVATE;

    // 写入查询结果
    uint64_t return_value = 0;
    if (lpBuffer != 0 && dwLength >= sizeof(MEMORY_BASIC_INFORMATION)) {
        uc_mem_write(uc, lpBuffer, &mbi, sizeof(MEMORY_BASIC_INFORMATION));
        return_value = sizeof(MEMORY_BASIC_INFORMATION);
    }

    // 设置返回值
    if (context->GetPeInfo()->isX64) {
        uc_reg_write(uc, UC_X86_REG_RAX, &return_value);
    } else {
        uint32_t return_value_32 = static_cast<uint32_t>(return_value);
        uc_reg_write(uc, UC_X86_REG_EAX, &return_value_32);
    }
}

auto Api_Process32FirstW(void* sandbox, uc_engine* uc, uint64_t address)
    -> void {
    auto* context = static_cast<Sandbox*>(sandbox);
    uint64_t hSnapshot = 0;
    uint64_t lppe = 0;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        // x64: rcx = hSnapshot, rdx = lppe (LPPROCESSENTRY32W)
        uc_reg_read(uc, UC_X86_REG_RCX, &hSnapshot);
        uc_reg_read(uc, UC_X86_REG_RDX, &lppe);
    } else {
        // x86: 从栈上读取参数
        uint32_t esp = 0;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp);
        esp += 0x4;  // 跳过返回地址
        uint32_t temp_handle;
        uint32_t temp_lppe;
        uc_mem_read(uc, esp, &temp_handle, sizeof(uint32_t));
        uc_mem_read(uc, esp + 0x4, &temp_lppe, sizeof(uint32_t));
        hSnapshot = temp_handle;
        lppe = temp_lppe;
    }

    // 验证句柄
    bool success = false;
    if (hSnapshot == 0x1337) {  // 我们在CreateToolhelp32Snapshot中使用的魔数
        // 读取结构体大小
        DWORD structSize = 0;
        if (uc_mem_read(uc, lppe, &structSize, sizeof(DWORD)) == UC_ERR_OK) {
            if (context->GetPeInfo()->isX64) {
                if (structSize == sizeof(PROCESSENTRY32W)) {
                    // 获取第一个进程信息（在我们的实现中是DingTalk.exe）
                    PROCESSENTRY32W pe32 = { 0 };
                    pe32.dwSize = sizeof(PROCESSENTRY32W);
                    pe32.th32ProcessID = 1001;  // DingTalk的PID
                    pe32.cntThreads = 1;
                    pe32.th32ParentProcessID = 4;  // 父进程是System
                    pe32.pcPriClassBase = 8;       // 正常优先级

                    // 设置进程名
                    std::wstring procName = L"DingTalk.exe";
                    wcscpy_s(pe32.szExeFile, procName.c_str());

                    // 写入进程信息到用户提供的缓冲区
                    if (uc_mem_write(uc, lppe, &pe32, sizeof(PROCESSENTRY32W)) ==
                        UC_ERR_OK) {
                        success = true;
                    }
                }
            }
            else {
                if (structSize == sizeof(PROCESSENTRY32W_32)) {
                    // 获取第一个进程信息（在我们的实现中是DingTalk.exe）
                    PROCESSENTRY32W_32 pe32 = { 0 };
                    pe32.dwSize = sizeof(PROCESSENTRY32W_32);
                    pe32.th32ProcessID = 1001;  // DingTalk的PID
                    pe32.cntThreads = 1;
                    pe32.th32ParentProcessID = 4;  // 父进程是System
                    pe32.pcPriClassBase = 8;       // 正常优先级

                    // 设置进程名
                    std::wstring procName = L"DingTalk.exe";
                    wcscpy_s(pe32.szExeFile, procName.c_str());

                    // 写入进程信息到用户提供的缓冲区
                    if (uc_mem_write(uc, lppe, &pe32, sizeof(PROCESSENTRY32W_32)) ==
                        UC_ERR_OK) {
                        success = true;
                    }
                }
            }
        }
    }

    printf("[*] Process32FirstW: Handle=0x%llx, Buffer=0x%llx, Success=%d\n",
           hSnapshot, lppe, success);

    // 设置返回值
    uint64_t result = success ? 1 : 0;
    if (context->GetPeInfo()->isX64) {
        uc_reg_write(uc, UC_X86_REG_RAX, &result);
    } else {
        uint32_t result32 = static_cast<uint32_t>(result);
        uc_reg_write(uc, UC_X86_REG_EAX, &result32);
    }

    // 设置错误码
    DWORD error = success ? 0 : ERROR_NO_MORE_FILES;
    if (context->GetPeInfo()->isX64) {
        context->GetTeb64()->LastErrorValue = error;
    } else {
        context->GetTeb32()->LastErrorValue = error;
    }
}
auto Api_CreateToolhelp32Snapshot(void* sandbox, uc_engine* uc,
                                  uint64_t address) -> void {
    auto* context = static_cast<Sandbox*>(sandbox);
    uint32_t dwFlags = 0;
    uint32_t th32ProcessID = 0;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        // x64: rcx = dwFlags, rdx = th32ProcessID
        uint64_t temp_flags;
        uint64_t temp_pid;
        uc_reg_read(uc, UC_X86_REG_RCX, &temp_flags);
        uc_reg_read(uc, UC_X86_REG_RDX, &temp_pid);
        dwFlags = static_cast<uint32_t>(temp_flags);
        th32ProcessID = static_cast<uint32_t>(temp_pid);
    } else {
        // x86: 从栈上读取参数
        uint32_t esp = 0;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp);
        esp += 0x4;  // 跳过返回地址
        uc_mem_read(uc, esp, &dwFlags, sizeof(uint32_t));
        uc_mem_read(uc, esp + 0x4, &th32ProcessID, sizeof(uint32_t));
    }

    // 创建一个假的句柄值
    uint64_t handle = 0x1337;

    // 如果请求进程列表快照，初始化进程枚举状态
    if (dwFlags & TH32CS_SNAPPROCESS) {
        // 初始化进程枚举状态为-1，这样Process32First会返回第一个进程
        context->process_enum_state[handle] = -1;

        // 清除错误码
        if (context->GetPeInfo()->isX64) {
            context->GetTeb64()->LastErrorValue = 0;
        } else {
            context->GetTeb32()->LastErrorValue = 0;
        }
    }

    printf(
        "[*] CreateToolhelp32Snapshot: Flags=0x%x, ProcessID=0x%x, "
        "Handle=0x%llx\n",
        dwFlags, th32ProcessID, handle);

    // 返回句柄
    if (context->GetPeInfo()->isX64) {
        uc_reg_write(uc, UC_X86_REG_RAX, &handle);
    } else {
        uint32_t handle32 = static_cast<uint32_t>(handle);
        uc_reg_write(uc, UC_X86_REG_EAX, &handle32);
    }
}
auto Api_Process32NextW(void* sandbox, uc_engine* uc, uint64_t address)
    -> void {
    auto* context = static_cast<Sandbox*>(sandbox);
    uint64_t hSnapshot = 0;
    uint64_t lppe = 0;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        // x64: rcx = hSnapshot, rdx = lppe (LPPROCESSENTRY32W)
        uc_reg_read(uc, UC_X86_REG_RCX, &hSnapshot);
        uc_reg_read(uc, UC_X86_REG_RDX, &lppe);
    } else {
        // x86: 从栈上读取参数
        uint32_t esp = 0;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp);
        esp += 0x4;  // 跳过返回地址
        uint32_t temp_handle;
        uint32_t temp_lppe;
        uc_mem_read(uc, esp, &temp_handle, sizeof(uint32_t));
        uc_mem_read(uc, esp + 0x4, &temp_lppe, sizeof(uint32_t));
        hSnapshot = temp_handle;
        lppe = temp_lppe;
    }
    // 获取当前进程索引
    size_t currentIndex = 0;
    auto it = context->process_enum_state.find(hSnapshot);
    if (it != context->process_enum_state.end()) {
        currentIndex = it->second;
        currentIndex++;  // 移动到下一个进程
    }

    // 定义进程列表
    struct ProcessInfo {
        const wchar_t* name;
        uint32_t pid;
        uint32_t parentPid;
    };

    ProcessInfo processes[] = {
        {L"DingTalk.exe", 1001, 4},   // 钉钉
        {L"Lanxin.exe", 1002, 4},     // 蓝信
        {L"QQ.exe", 1003, 4},         // QQ
        {L"Feishu.exe", 1004, 4},     // 飞书
        {L"explorer.exe", 1005, 4},   // Windows 资源管理器
        {L"svchost.exe", 1006, 4},    // 系统服务宿主进程
        {L"System", 4, 0},            // 系统进程
        {L"smss.exe", 376, 4},        // 会话管理器
        {L"csrss.exe", 648, 376},     // 客户端服务器运行时子系统
        {L"winlogon.exe", 672, 376},  // Windows 登录进程
    };

    const size_t processCount =
        sizeof(processes) / sizeof(processes[0]);

    // 验证句柄
    bool success = false;
    if (hSnapshot == 0x1337) {  // 我们在CreateToolhelp32Snapshot中使用的魔数
        // 读取结构体大小
        DWORD structSize = 0;
        if (uc_mem_read(uc, lppe, &structSize, sizeof(DWORD)) == UC_ERR_OK) {
            if (context->GetPeInfo()->isX64) {
                if (structSize == sizeof(PROCESSENTRY32W)) {
                    // 检查是否还有更多进程
                    if (currentIndex < processCount) {
                        // 填充进程信息
                        PROCESSENTRY32W pe32 = { 0 };
                        pe32.dwSize = sizeof(PROCESSENTRY32W);
                        pe32.th32ProcessID = processes[currentIndex].pid;
                        pe32.th32ParentProcessID =
                            processes[currentIndex].parentPid;
                        pe32.cntThreads = 1;
                        pe32.pcPriClassBase = 8;  // 正常优先级

                        // 设置进程名
                        wcscpy_s(pe32.szExeFile, processes[currentIndex].name);

                        // 写入进程信息到用户提供的缓冲区
                        if (uc_mem_write(uc, lppe, &pe32,
                            sizeof(PROCESSENTRY32W)) == UC_ERR_OK) {
                            success = true;
                            // 更新进程索引
                            context->process_enum_state[hSnapshot] = currentIndex;
                        }
                    }
                }
            }
            else {
                if (currentIndex < processCount) {
                    // 填充进程信息
                    PROCESSENTRY32W_32 pe32 = { 0 };
                    pe32.dwSize = sizeof(PROCESSENTRY32W_32);
                    pe32.th32ProcessID = processes[currentIndex].pid;
                    pe32.th32ParentProcessID =
                        processes[currentIndex].parentPid;
                    pe32.cntThreads = 1;
                    pe32.pcPriClassBase = 8;  // 正常优先级

                    // 设置进程名
                    wcscpy_s(pe32.szExeFile, processes[currentIndex].name);

                    // 写入进程信息到用户提供的缓冲区
                    if (uc_mem_write(uc, lppe, &pe32,
                        sizeof(PROCESSENTRY32W_32)) == UC_ERR_OK) {
                        success = true;
                        // 更新进程索引
                        context->process_enum_state[hSnapshot] = currentIndex;
                    }
                }
            }
        }
    }

    printf("[*] Process32NextW: Handle=0x%llx, Buffer=0x%llx, Success=%d\n",
           hSnapshot, lppe, success);

    // 设置返回值
    uint64_t result = success ? 1 : 0;
    if (context->GetPeInfo()->isX64) {
        uc_reg_write(uc, UC_X86_REG_RAX, &result);
    } else {
        uint32_t result32 = static_cast<uint32_t>(result);
        uc_reg_write(uc, UC_X86_REG_EAX, &result32);
    }

    // 设置错误码
    DWORD error = success ? 0 : ERROR_NO_MORE_FILES;
    if (context->GetPeInfo()->isX64) {
        context->GetTeb64()->LastErrorValue = error;
    } else {
        context->GetTeb32()->LastErrorValue = error;
    }
}

// 内部函数，用于处理CreateProcessA和CreateProcessW的共同逻辑
auto CreateProcessInternal(void* sandbox, uc_engine* uc,
                           uint64_t lpApplicationName, uint64_t lpCommandLine,
                           uint64_t lpProcessInformation,
                           uint64_t lpStartupInfo, bool isWideChar) -> bool {
    auto* context = static_cast<Sandbox*>(sandbox);
    std::string applicationName;
    std::wstring wApplicationName;
    std::string commandLine;
    std::wstring wCommandLine;

    // 读取应用程序名称
    if (lpApplicationName != 0) {
        if (isWideChar) {
            // 读取宽字符应用程序名称
            wchar_t buffer[MAX_PATH] = {0};
            size_t i = 0;
            bool success = true;

            do {
                wchar_t ch;
                uc_err err =
                    uc_mem_read(uc, lpApplicationName + (i * 2), &ch, 2);
                if (err != UC_ERR_OK) {
                    printf(
                        "[!] Error reading wide application name at address "
                        "0x%llx: %u\n",
                        lpApplicationName + (i * 2), err);
                    success = false;
                    break;
                }
                if (ch == 0) break;
                buffer[i] = ch;
                i++;
            } while (i < MAX_PATH - 1);

            // 确保字符串以 NULL 结尾
            buffer[i] = 0;

            if (success) {
                wApplicationName = std::wstring(buffer);
                // 转换为ANSI以便于日志记录
                std::string ansiAppName(wApplicationName.begin(),
                                        wApplicationName.end());
                printf("[*] Read Wide ApplicationName: %s (Length: %zu)\n",
                       ansiAppName.c_str(), wApplicationName.length());
            }
        } else {
            // 读取ANSI应用程序名称
            char buffer[MAX_PATH] = {0};
            size_t i = 0;
            bool success = true;

            do {
                uint8_t byte;
                uc_err err = uc_mem_read(uc, lpApplicationName + i, &byte, 1);
                if (err != UC_ERR_OK) {
                    printf(
                        "[!] Error reading application name at address 0x%llx: "
                        "%u\n",
                        lpApplicationName + i, err);
                    success = false;
                    break;
                }
                if (byte == 0) break;
                buffer[i] = byte;
                i++;
            } while (i < MAX_PATH - 1);

            // 确保字符串以 NULL 结尾
            buffer[i] = 0;

            if (success) {
                applicationName = std::string(buffer);
                printf("[*] Read ANSI ApplicationName: %s (Length: %zu)\n",
                       buffer, applicationName.length());
                // 转换为宽字符
                wApplicationName = std::wstring(applicationName.begin(),
                                                applicationName.end());
            }
        }
    }

    // 读取命令行
    if (lpCommandLine != 0) {
        if (isWideChar) {
            // 读取宽字符命令行
            wchar_t buffer[MAX_PATH] = {0};
            size_t i = 0;
            bool success = true;

            do {
                wchar_t ch;
                uc_err err = uc_mem_read(uc, lpCommandLine + (i * 2), &ch, 2);
                if (err != UC_ERR_OK) {
                    printf(
                        "[!] Error reading wide command line at address "
                        "0x%llx: %u\n",
                        lpCommandLine + (i * 2), err);
                    success = false;
                    break;
                }
                if (ch == 0) break;
                buffer[i] = ch;
                i++;
            } while (i < MAX_PATH - 1);

            // 确保字符串以 NULL 结尾
            buffer[i] = 0;

            if (success) {
                wCommandLine = std::wstring(buffer);
                // 转换为ANSI以便于日志记录
                std::string ansiCmdLine(wCommandLine.begin(),
                                        wCommandLine.end());
                printf("[*] Read Wide CommandLine: %s (Length: %zu)\n",
                       ansiCmdLine.c_str(), wCommandLine.length());
            }
        } else {
            // 读取ANSI命令行
            char buffer[MAX_PATH] = {0};
            size_t i = 0;
            bool success = true;

            do {
                uint8_t byte;
                uc_err err = uc_mem_read(uc, lpCommandLine + i, &byte, 1);
                if (err != UC_ERR_OK) {
                    printf(
                        "[!] Error reading command line at address 0x%llx: "
                        "%u\n",
                        lpCommandLine + i, err);
                    success = false;
                    break;
                }
                if (byte == 0) break;
                buffer[i] = byte;
                i++;
            } while (i < MAX_PATH - 1);

            // 确保字符串以 NULL 结尾
            buffer[i] = 0;

            if (success) {
                commandLine = std::string(buffer);
                printf("[*] Read ANSI CommandLine: %s (Length: %zu)\n", buffer,
                       commandLine.length());
                // 转换为宽字符
                wCommandLine =
                    std::wstring(commandLine.begin(), commandLine.end());
            }
        }
    }

    // 记录调用信息
    if (isWideChar) {
        std::string ansiAppName(wApplicationName.begin(),
                                wApplicationName.end());
        std::string ansiCmdLine(wCommandLine.begin(), wCommandLine.end());
        printf("[*] CreateProcessW: ApplicationName=%s, CommandLine=%s\n",
               ansiAppName.empty() ? "(null)" : ansiAppName.c_str(),
               ansiCmdLine.empty() ? "(null)" : ansiCmdLine.c_str());
    } else {
        printf("[*] CreateProcessA: ApplicationName=%s, CommandLine=%s\n",
               applicationName.empty() ? "(null)" : applicationName.c_str(),
               commandLine.empty() ? "(null)" : commandLine.c_str());
    }

    // 模拟创建进程，设置进程和线程ID
    DWORD processId = 0x1234;
    DWORD threadId = 0x5678;
    HANDLE hProcess = (HANDLE)0x1340;
    HANDLE hThread = (HANDLE)0x1341;

    // 写入进程信息
    if (lpProcessInformation != 0) {
        if (context->GetPeInfo()->isX64) {
            struct PROCESS_INFORMATION64 {
                HANDLE hProcess;
                HANDLE hThread;
                DWORD dwProcessId;
                DWORD dwThreadId;
            } pi;
            pi.hProcess = (HANDLE)hProcess;
            pi.hThread = (HANDLE)hThread;
            pi.dwProcessId = processId;
            pi.dwThreadId = threadId;
            uc_mem_write(uc, lpProcessInformation, &pi, sizeof(pi));
            printf("[*] Wrote process info (x64) to 0x%llx\n",
                   lpProcessInformation);
        } else {
            struct _PROCESS_INFORMATION32 {
                DWORD hProcess;
                DWORD hThread;
                DWORD dwProcessId;
                DWORD dwThreadId;
            } pi;
            pi.hProcess = (DWORD)hProcess;
            pi.hThread = (DWORD)hThread;
            pi.dwProcessId = (DWORD)processId;
            pi.dwThreadId = threadId;
            uc_mem_write(uc, lpProcessInformation, &pi, sizeof(pi));
            printf("[*] Wrote process info (x86) to 0x%llx\n",
                   lpProcessInformation);
        }
    }

    return true;
}

auto Api_CreateProcessA(void* sandbox, uc_engine* uc, uint64_t address)
    -> void {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t lpApplicationName = 0;
    uint64_t lpCommandLine = 0;
    uint64_t lpProcessInformation = 0;
    uint64_t lpStartupInfo = 0;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        // x64: rcx = lpApplicationName, rdx = lpCommandLine
        uc_reg_read(uc, UC_X86_REG_RCX, &lpApplicationName);
        uc_reg_read(uc, UC_X86_REG_RDX, &lpCommandLine);
        // 从栈上获取 PROCESS_INFORMATION 和 STARTUPINFO
        uint64_t rsp;
        uc_reg_read(uc, UC_X86_REG_RSP, &rsp);
        rsp += 0x28;  // 跳过前4个参数的影子空间
        uc_mem_read(uc, rsp + 0x20, &lpProcessInformation, sizeof(uint64_t));
        uc_mem_read(uc, rsp + 0x18, &lpStartupInfo, sizeof(uint64_t));

        printf(
            "[*] CreateProcessA Debug (x64): AppNameAddr=0x%llx, "
            "CmdLineAddr=0x%llx\n",
            lpApplicationName, lpCommandLine);
    } else {
        // x86: 从栈上读取参数
        uint32_t esp;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp);
        esp += 0x4;  // 跳过返回地址
        uint32_t temp_app_name, temp_cmd_line, temp_proc_info,
            temp_startup_info;
        uc_mem_read(uc, esp, &temp_app_name, sizeof(uint32_t));
        uc_mem_read(uc, esp + 0x4, &temp_cmd_line, sizeof(uint32_t));
        // 修正x86下的参数读取偏移，使用实际结构的偏移量
        uc_mem_read(uc, esp + 0x24, &temp_proc_info, sizeof(uint32_t));
        uc_mem_read(uc, esp + 0x1C, &temp_startup_info, sizeof(uint32_t));
        lpApplicationName = temp_app_name;
        lpCommandLine = temp_cmd_line;
        lpProcessInformation = temp_proc_info;
        lpStartupInfo = temp_startup_info;

        printf(
            "[*] CreateProcessA Debug (x86): ESP=0x%x, AppNameAddr=0x%x, "
            "CmdLineAddr=0x%x\n",
            esp, temp_app_name, temp_cmd_line);
    }

    // 调用内部函数处理共同逻辑
    bool success =
        CreateProcessInternal(sandbox, uc, lpApplicationName, lpCommandLine,
                              lpProcessInformation, lpStartupInfo, false);

    // 返回结果
    uint64_t result = success ? 1 : 0;
    uc_reg_write(uc,
                 context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
                 &result);
}

// CreateProcessW的实现
auto Api_CreateProcessW(void* sandbox, uc_engine* uc, uint64_t address)
    -> void {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t lpApplicationName = 0;
    uint64_t lpCommandLine = 0;
    uint64_t lpProcessInformation = 0;
    uint64_t lpStartupInfo = 0;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        // x64: rcx = lpApplicationName, rdx = lpCommandLine
        uc_reg_read(uc, UC_X86_REG_RCX, &lpApplicationName);
        uc_reg_read(uc, UC_X86_REG_RDX, &lpCommandLine);
        // 从栈上获取 PROCESS_INFORMATION 和 STARTUPINFO
        uint64_t rsp;
        uc_reg_read(uc, UC_X86_REG_RSP, &rsp);
        rsp += 0x28;  // 跳过前4个参数的影子空间
        uc_mem_read(uc, rsp + 0x20, &lpProcessInformation, sizeof(uint64_t));
        uc_mem_read(uc, rsp + 0x18, &lpStartupInfo, sizeof(uint64_t));

        printf(
            "[*] CreateProcessW Debug (x64): AppNameAddr=0x%llx, "
            "CmdLineAddr=0x%llx\n",
            lpApplicationName, lpCommandLine);
    } else {
        // x86: 从栈上读取参数
        uint32_t esp;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp);
        esp += 0x4;  // 跳过返回地址
        uint32_t temp_app_name, temp_cmd_line, temp_proc_info,
            temp_startup_info;
        uc_mem_read(uc, esp, &temp_app_name, sizeof(uint32_t));
        uc_mem_read(uc, esp + 0x4, &temp_cmd_line, sizeof(uint32_t));
        // 修正x86下的参数读取偏移，使用实际结构的偏移量
        uc_mem_read(uc, esp + 0x24, &temp_proc_info, sizeof(uint32_t));
        uc_mem_read(uc, esp + 0x1C, &temp_startup_info, sizeof(uint32_t));
        lpApplicationName = temp_app_name;
        lpCommandLine = temp_cmd_line;
        lpProcessInformation = temp_proc_info;
        lpStartupInfo = temp_startup_info;

        printf(
            "[*] CreateProcessW Debug (x86): ESP=0x%x, AppNameAddr=0x%x, "
            "CmdLineAddr=0x%x\n",
            esp, temp_app_name, temp_cmd_line);
    }

    // 调用内部函数处理共同逻辑
    bool success =
        CreateProcessInternal(sandbox, uc, lpApplicationName, lpCommandLine,
                              lpProcessInformation, lpStartupInfo, true);

    // 返回结果
    uint64_t result = success ? 1 : 0;
    uc_reg_write(uc,
                 context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
                 &result);
}

auto Api_GetCurrentProcess(void* sandbox, uc_engine* uc, uint64_t address)
    -> void {
    auto context = static_cast<Sandbox*>(sandbox);

    // GetCurrentProcess 总是返回伪句柄值 -1 (0xFFFFFFFF)
    uint64_t pseudo_handle = static_cast<uint64_t>(-1);

    // 根据架构写入返回值
    uc_reg_write(uc,
                 context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
                 &pseudo_handle);

    printf("[*] GetCurrentProcess called, returning pseudo-handle 0x%llx\n",
           pseudo_handle);
}

auto Api_OpenProcessToken(void* sandbox, uc_engine* uc, uint64_t address)
    -> void {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t ProcessHandle = 0;
    uint32_t DesiredAccess = 0;
    uint64_t TokenHandle = 0;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        // x64: rcx = ProcessHandle, rdx = DesiredAccess, r8 = TokenHandle
        uc_reg_read(uc, UC_X86_REG_RCX, &ProcessHandle);
        uint64_t temp_access;
        uc_reg_read(uc, UC_X86_REG_RDX, &temp_access);
        DesiredAccess = static_cast<uint32_t>(temp_access);
        uc_reg_read(uc, UC_X86_REG_R8, &TokenHandle);
    } else {
        // x86: 从栈上读取参数
        uint32_t esp_address = 0;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp_address);
        esp_address += 0x4;  // 跳过返回地址

        uint32_t temp_handle;
        uc_mem_read(uc, esp_address, &temp_handle, sizeof(uint32_t));
        ProcessHandle = temp_handle;
        esp_address += 0x4;

        uc_mem_read(uc, esp_address, &DesiredAccess, sizeof(uint32_t));
        esp_address += 0x4;

        uint32_t temp_token;
        uc_mem_read(uc, esp_address, &temp_token, sizeof(uint32_t));
        TokenHandle = temp_token;
    }

    // 创建一个假的token句柄（使用一个非零值）
    uint64_t fake_token_handle = 0x1234;

    // 将假的token句柄写入TokenHandle指向的内存
    if (TokenHandle != 0) {
        if (context->GetPeInfo()->isX64) {
            uc_mem_write(uc, TokenHandle, &fake_token_handle, sizeof(uint64_t));
        } else {
            uint32_t token_handle_32 = static_cast<uint32_t>(fake_token_handle);
            uc_mem_write(uc, TokenHandle, &token_handle_32, sizeof(uint32_t));
        }
    }

    // 返回TRUE
    uint64_t result = 1;
    uc_reg_write(uc,
                 context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
                 &result);

    printf(
        "[*] OpenProcessToken: ProcessHandle=0x%llx, DesiredAccess=0x%x, "
        "TokenHandle=0x%llx\n",
        ProcessHandle, DesiredAccess, fake_token_handle);
}

auto Api_GetTokenInformation(void* sandbox, uc_engine* uc, uint64_t address)
    -> void {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t TokenHandle = 0;
    uint32_t TokenInformationClass = 0;
    uint64_t TokenInformation = 0;
    uint32_t TokenInformationLength = 0;
    uint64_t ReturnLength = 0;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        // x64: rcx, rdx, r8, r9, [rsp+0x28]
        uc_reg_read(uc, UC_X86_REG_RCX, &TokenHandle);
        uint64_t temp_class;
        uc_reg_read(uc, UC_X86_REG_RDX, &temp_class);
        TokenInformationClass = static_cast<uint32_t>(temp_class);
        uc_reg_read(uc, UC_X86_REG_R8, &TokenInformation);
        uint64_t temp_length;
        uc_reg_read(uc, UC_X86_REG_R9, &temp_length);
        TokenInformationLength = static_cast<uint32_t>(temp_length);

        uint64_t rsp;
        uc_reg_read(uc, UC_X86_REG_RSP, &rsp);
        uc_mem_read(uc, rsp + 0x28, &ReturnLength, sizeof(uint64_t));
    } else {
        // x86: 从栈上读取参数
        uint32_t esp_address = 0;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp_address);
        esp_address += 0x4;  // 跳过返回地址

        uint32_t temp_handle;
        uc_mem_read(uc, esp_address, &temp_handle, sizeof(uint32_t));
        TokenHandle = temp_handle;
        esp_address += 0x4;

        uc_mem_read(uc, esp_address, &TokenInformationClass, sizeof(uint32_t));
        esp_address += 0x4;

        uint32_t temp_info;
        uc_mem_read(uc, esp_address, &temp_info, sizeof(uint32_t));
        TokenInformation = temp_info;
        esp_address += 0x4;

        uc_mem_read(uc, esp_address, &TokenInformationLength, sizeof(uint32_t));
        esp_address += 0x4;

        uint32_t temp_return;
        uc_mem_read(uc, esp_address, &temp_return, sizeof(uint32_t));
        ReturnLength = temp_return;
    }

    // 如果是TokenElevation类（20），返回TRUE表示进程有管理员权限
    if (TokenInformationClass == 20) {  // TokenElevation
        uint32_t is_elevated = 1;       // 1表示有管理员权限
        if (TokenInformation != 0 &&
            TokenInformationLength >= sizeof(uint32_t)) {
            uc_mem_write(uc, TokenInformation, &is_elevated, sizeof(uint32_t));
        }

        // 写入所需的缓冲区大小
        uint32_t required_size = sizeof(uint32_t);
        if (ReturnLength != 0) {
            uc_mem_write(uc, ReturnLength, &required_size, sizeof(uint32_t));
        }
    }

    // 返回TRUE
    uint64_t result = 1;
    uc_reg_write(uc,
                 context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
                 &result);

    printf(
        "[*] GetTokenInformation: TokenHandle=0x%llx, Class=%d, Info=0x%llx, "
        "Length=%u\n",
        TokenHandle, TokenInformationClass, TokenInformation,
        TokenInformationLength);
}
