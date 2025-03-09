#include "sandbox.h"
#include "sandbox_callbacks.h"

auto Api_RegOpenKeyExW(void* sandbox, uc_engine* uc, uint64_t address) -> void {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t hKey = 0;        // 父键句柄
    uint64_t lpSubKey = 0;    // 子键名称
    uint32_t ulOptions = 0;   // 选项
    uint32_t samDesired = 0;  // 访问权限
    uint64_t phkResult = 0;   // 结果句柄的指针

    // 默认返回值：成功
    LONG status = ERROR_SUCCESS;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        // x64: rcx=hKey, rdx=lpSubKey, r8=ulOptions, r9=samDesired,
        // [rsp+0x28]=phkResult
        uc_reg_read(uc, UC_X86_REG_RCX, &hKey);
        uc_reg_read(uc, UC_X86_REG_RDX, &lpSubKey);
        uint64_t temp_options = 0;
        uc_reg_read(uc, UC_X86_REG_R8, &temp_options);
        ulOptions = static_cast<uint32_t>(temp_options);
        uint64_t temp_sam = 0;
        uc_reg_read(uc, UC_X86_REG_R9, &temp_sam);
        samDesired = static_cast<uint32_t>(temp_sam);

        // 第5个参数从栈上读取
        uint64_t rsp = 0;
        uc_reg_read(uc, UC_X86_REG_RSP, &rsp);
        uc_mem_read(uc, rsp + 0x28, &phkResult, sizeof(uint64_t));
    } else {
        // x86: 从栈上读取参数
        uint32_t esp_address = 0;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp_address);
        esp_address += 4;  // 跳过返回地址

        uint32_t temp_hkey = 0;
        uc_mem_read(uc, esp_address, &temp_hkey, sizeof(uint32_t));
        hKey = temp_hkey;
        esp_address += 4;

        uint32_t temp_subkey = 0;
        uc_mem_read(uc, esp_address, &temp_subkey, sizeof(uint32_t));
        lpSubKey = temp_subkey;
        esp_address += 4;

        uc_mem_read(uc, esp_address, &ulOptions, sizeof(uint32_t));
        esp_address += 4;

        uc_mem_read(uc, esp_address, &samDesired, sizeof(uint32_t));
        esp_address += 4;

        uint32_t temp_result = 0;
        uc_mem_read(uc, esp_address, &temp_result, sizeof(uint32_t));
        phkResult = temp_result;
    }

    // 读取子键名称
    std::wstring subKeyName;
    if (lpSubKey != 0) {
        wchar_t buffer[MAX_PATH] = {0};
        size_t bytesRead = 0;
        bool truncated = false;

        // 读取Unicode字符串，直到遇到null终止符或达到MAX_PATH
        for (size_t i = 0; i < MAX_PATH - 1; i++) {
            wchar_t ch = 0;
            uc_mem_read(uc, lpSubKey + (i * sizeof(wchar_t)), &ch,
                        sizeof(wchar_t));
            if (ch == 0) break;
            buffer[i] = ch;
            bytesRead = i + 1;

            if (i == MAX_PATH - 2) {
                truncated = true;
            }
        }

        subKeyName = std::wstring(buffer, bytesRead);
    }

    // 生成一个随机句柄值 (不是0，通常是4的倍数)
    uint32_t newKeyHandle = 0x1000 + (std::rand() % 0xFFFFF) * 4;

    // 在沙箱中记录打开的注册表键 (这里可以根据需要扩展，保存键的路径等信息)
    // 例如：context->OpenedRegistryKeys[newKeyHandle] = {hKey, subKeyName};

    // 写入句柄到结果指针
    if (phkResult != 0) {
        if (context->GetPeInfo()->isX64) {
            uc_mem_write(uc, phkResult, &newKeyHandle, sizeof(uint64_t));
        } else {
            uc_mem_write(uc, phkResult, &newKeyHandle, sizeof(uint32_t));
        }
    } else {
        status = ERROR_INVALID_PARAMETER;
    }
    // 获取根键名称
    std::string rootKeyName;
    switch (hKey) {
        case (uint64_t)HKEY_CLASSES_ROOT:
            rootKeyName = "HKEY_CLASSES_ROOT";
            break;
        case (uint64_t)HKEY_CURRENT_USER:
            rootKeyName = "HKEY_CURRENT_USER";
            break;
        case (uint64_t)HKEY_LOCAL_MACHINE:
            rootKeyName = "HKEY_LOCAL_MACHINE";
            break;
        case (uint64_t)HKEY_USERS:
            rootKeyName = "HKEY_USERS";
            break;
        case (uint64_t)HKEY_CURRENT_CONFIG:
            rootKeyName = "HKEY_CURRENT_CONFIG";
            break;
        default:
            rootKeyName = "Unknown key handle";
            break;
    }

    std::string wstr_to_str;
    for (wchar_t c : subKeyName) {
        if (c <= 127) {
            wstr_to_str += static_cast<char>(c);
        } else {
            wstr_to_str += '?';
        }
    }
    context->CheckMalwareActive_Registry(subKeyName);

    printf(
        "[*] RegOpenKeyExW: %s\\%s, Options=0x%x, SAM=0x%x -> Handle=0x%x, "
        "Status=%ld\n",
        rootKeyName.c_str(), wstr_to_str.c_str(), ulOptions, samDesired,
        newKeyHandle, status);

    // 返回状态
    uc_reg_write(uc,
                 context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
                 &status);
}

auto Api_RegCloseKey(void* sandbox, uc_engine* uc, uint64_t address) -> void {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t hKey = 0;  // 键句柄

    // 默认返回值：成功
    LONG status = ERROR_SUCCESS;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        // x64: rcx=hKey
        uc_reg_read(uc, UC_X86_REG_RCX, &hKey);
    } else {
        // x86: 从栈上读取参数
        uint32_t esp_address = 0;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp_address);
        esp_address += 4;  // 跳过返回地址

        uint32_t temp_hkey = 0;
        uc_mem_read(uc, esp_address, &temp_hkey, sizeof(uint32_t));
        hKey = temp_hkey;
    }

    // 在实际应用中，这里应该从沙盒的注册表句柄映射中移除此句柄
    // 但当前环境似乎没有明确保存句柄映射，所以只记录操作即可
    // 如果以后需要，可以添加: context->OpenedRegistryKeys.erase(hKey);

    // 只有当句柄为0或无效时才返回错误
    if (hKey == 0) {
        status = ERROR_INVALID_HANDLE;
    }

    printf("[*] RegCloseKey: Handle=0x%llx -> Status=%ld\n", hKey, status);

    // 返回状态
    uc_reg_write(uc,
                 context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
                 &status);
}
