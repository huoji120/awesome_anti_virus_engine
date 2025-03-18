#include "sandbox.h"
#include "sandbox_callbacks.h"
#include "sandbox_api_winhttp.h"
#include <tlhelp32.h>

// WLAN API 实现
auto Api_WlanOpenHandle(void* sandbox, uc_engine* uc, uint64_t address)
    -> void {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t dwClientVersion = 0;
    uint64_t pReserved = 0;
    uint64_t pdwNegotiatedVersion = 0;
    uint64_t phClientHandle = 0;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        uc_reg_read(uc, UC_X86_REG_RCX, &dwClientVersion);
        uc_reg_read(uc, UC_X86_REG_RDX, &pReserved);
        uc_reg_read(uc, UC_X86_REG_R8, &pdwNegotiatedVersion);
        uc_reg_read(uc, UC_X86_REG_R9, &phClientHandle);
    } else {
        uint32_t esp;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp);
        esp += 4;
        uc_mem_read(uc, esp, &dwClientVersion, sizeof(uint32_t));
        esp += 4;
        uint32_t temp_reserved;
        uc_mem_read(uc, esp, &temp_reserved, sizeof(uint32_t));
        pReserved = temp_reserved;
        esp += 4;
        uint32_t temp_version;
        uc_mem_read(uc, esp, &temp_version, sizeof(uint32_t));
        pdwNegotiatedVersion = temp_version;
        esp += 4;
        uint32_t temp_handle;
        uc_mem_read(uc, esp, &temp_handle, sizeof(uint32_t));
        phClientHandle = temp_handle;
    }

    // 修改常量定义
    uint32_t negotiatedVersion = 2;      // 返回请求的版本
    uint64_t clientHandle = 0x13370000;  // 使用有效的十六进制常量

    // 写入协商版本
    if (pdwNegotiatedVersion != 0) {
        uc_mem_write(uc, pdwNegotiatedVersion, &negotiatedVersion,
                     sizeof(uint32_t));
    }

    // 写入客户端句柄
    if (phClientHandle != 0) {
        if (context->GetPeInfo()->isX64) {
            uc_mem_write(uc, phClientHandle, &clientHandle, sizeof(uint64_t));
        } else {
            uint32_t handle32 = static_cast<uint32_t>(clientHandle);
            uc_mem_write(uc, phClientHandle, &handle32, sizeof(uint32_t));
        }
    }

    // 返回成功（0）
    uint64_t result = 0;
    uc_reg_write(uc,
                 context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
                 &result);

    printf("[*] WlanOpenHandle: Version=%u, Handle=0x%llx\n", negotiatedVersion,
           clientHandle);
}

auto Api_WlanEnumInterfaces(void* sandbox, uc_engine* uc, uint64_t address)
    -> void {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t hClientHandle = 0;
    uint64_t pReserved = 0;
    uint64_t ppInterfaceList = 0;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        uc_reg_read(uc, UC_X86_REG_RCX, &hClientHandle);
        uc_reg_read(uc, UC_X86_REG_RDX, &pReserved);
        uc_reg_read(uc, UC_X86_REG_R8, &ppInterfaceList);
    } else {
        uint32_t esp;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp);
        esp += 4;
        uint32_t temp_handle;
        uc_mem_read(uc, esp, &temp_handle, sizeof(uint32_t));
        hClientHandle = temp_handle;
        esp += 4;
        uint32_t temp_reserved;
        uc_mem_read(uc, esp, &temp_reserved, sizeof(uint32_t));
        pReserved = temp_reserved;
        esp += 4;
        uint32_t temp_list;
        uc_mem_read(uc, esp, &temp_list, sizeof(uint32_t));
        ppInterfaceList = temp_list;
    }

    // 修改句柄检查
    if (hClientHandle != 0x13370000) {
        uint64_t result = 1;  // ERROR_INVALID_HANDLE
        uc_reg_write(
            uc, context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
            &result);
        return;
    }

    // 分配内存用于接口列表
    uint64_t interfaceListAddr = context->AllocateMemory(1024);  // 足够大的空间

    // 创建一个模拟的WLAN接口列表
    struct WLAN_INTERFACE_INFO {
        GUID InterfaceGuid;
        WCHAR strInterfaceDescription[256];
        DWORD isState;
    };

    struct WLAN_INTERFACE_INFO_LIST {
        DWORD dwNumberOfItems;
        DWORD dwIndex;
        WLAN_INTERFACE_INFO InterfaceInfo[1];
    };

    WLAN_INTERFACE_INFO_LIST interfaceList = {0};
    interfaceList.dwNumberOfItems = 1;
    interfaceList.dwIndex = 0;

    // 创建一个假的GUID
    GUID fakeGuid = {0x12345678,
                     0x1234,
                     0x1234,
                     {0x12, 0x34, 0x56, 0x78, 0x90, 0xAB, 0xCD, 0xEF}};
    interfaceList.InterfaceInfo[0].InterfaceGuid = fakeGuid;

    // 设置接口描述
    const wchar_t* description = L"Simulated Wi-Fi Adapter";
    wcscpy_s(interfaceList.InterfaceInfo[0].strInterfaceDescription,
             description);
    interfaceList.InterfaceInfo[0].isState = 1;  // connected

    // 写入接口列表
    uc_mem_write(uc, interfaceListAddr, &interfaceList,
                 sizeof(WLAN_INTERFACE_INFO_LIST));

    // 写入接口列表指针
    if (context->GetPeInfo()->isX64) {
        uc_mem_write(uc, ppInterfaceList, &interfaceListAddr, sizeof(uint64_t));
    } else {
        uint32_t addr32 = static_cast<uint32_t>(interfaceListAddr);
        uc_mem_write(uc, ppInterfaceList, &addr32, sizeof(uint32_t));
    }

    // 返回成功（0）
    uint64_t result = 0;
    uc_reg_write(uc,
                 context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
                 &result);

    printf("[*] WlanEnumInterfaces: Handle=0x%llx, InterfaceList=0x%llx\n",
           hClientHandle, interfaceListAddr);
}

auto Api_WlanGetProfileList(void* sandbox, uc_engine* uc, uint64_t address)
    -> void {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t hClientHandle = 0;
    uint64_t pInterfaceGuid = 0;
    uint64_t pReserved = 0;
    uint64_t ppProfileList = 0;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        uc_reg_read(uc, UC_X86_REG_RCX, &hClientHandle);
        uc_reg_read(uc, UC_X86_REG_RDX, &pInterfaceGuid);
        uc_reg_read(uc, UC_X86_REG_R8, &pReserved);
        uc_reg_read(uc, UC_X86_REG_R9, &ppProfileList);
    } else {
        uint32_t esp;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp);
        esp += 4;
        uint32_t temp_values[4];
        uc_mem_read(uc, esp, temp_values, sizeof(uint32_t) * 4);
        hClientHandle = temp_values[0];
        pInterfaceGuid = temp_values[1];
        pReserved = temp_values[2];
        ppProfileList = temp_values[3];
    }

    // 分配内存用于配置文件列表
    uint64_t profileListAddr = context->AllocateMemory(1024);

    // 创建模拟的配置文件列表
    struct WLAN_PROFILE_INFO {
        WCHAR strProfileName[256];
        DWORD dwFlags;
    };

    struct WLAN_PROFILE_INFO_LIST {
        DWORD dwNumberOfItems;
        DWORD dwIndex;
        WLAN_PROFILE_INFO ProfileInfo[1];
    };

    WLAN_PROFILE_INFO_LIST profileList = {0};
    profileList.dwNumberOfItems = 1;
    profileList.dwIndex = 0;

    // 设置一个模拟的配置文件
    const wchar_t* profileName = L"Home Network";
    wcscpy_s(profileList.ProfileInfo[0].strProfileName, profileName);
    profileList.ProfileInfo[0].dwFlags = 1;

    // 写入配置文件列表
    uc_mem_write(uc, profileListAddr, &profileList,
                 sizeof(WLAN_PROFILE_INFO_LIST));

    // 写入配置文件列表指针
    if (context->GetPeInfo()->isX64) {
        uc_mem_write(uc, ppProfileList, &profileListAddr, sizeof(uint64_t));
    } else {
        uint32_t addr32 = static_cast<uint32_t>(profileListAddr);
        uc_mem_write(uc, ppProfileList, &addr32, sizeof(uint32_t));
    }

    // 返回成功（0）
    uint64_t result = 0;
    uc_reg_write(uc,
                 context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
                 &result);

    printf("[*] WlanGetProfileList: Handle=0x%llx, ProfileList=0x%llx\n",
           hClientHandle, profileListAddr);
}

auto Api_WlanFreeMemory(void* sandbox, uc_engine* uc, uint64_t address)
    -> void {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t pMemory = 0;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        uc_reg_read(uc, UC_X86_REG_RCX, &pMemory);
    } else {
        uint32_t esp;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp);
        esp += 4;
        uint32_t temp_memory;
        uc_mem_read(uc, esp, &temp_memory, sizeof(uint32_t));
        pMemory = temp_memory;
    }

    // 实际上我们不需要释放内存，因为这是在模拟环境中
    printf("[*] WlanFreeMemory: Memory=0x%llx\n", pMemory);
}

auto Api_WlanCloseHandle(void* sandbox, uc_engine* uc, uint64_t address)
    -> void {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t hClientHandle = 0;
    uint64_t pReserved = 0;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        uc_reg_read(uc, UC_X86_REG_RCX, &hClientHandle);
        uc_reg_read(uc, UC_X86_REG_RDX, &pReserved);
    } else {
        uint32_t esp;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp);
        esp += 4;
        uint32_t temp_handle;
        uc_mem_read(uc, esp, &temp_handle, sizeof(uint32_t));
        hClientHandle = temp_handle;
        esp += 4;
        uint32_t temp_reserved;
        uc_mem_read(uc, esp, &temp_reserved, sizeof(uint32_t));
        pReserved = temp_reserved;
    }

    // 返回成功（0）
    uint64_t result = 0;
    uc_reg_write(uc,
                 context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
                 &result);

    printf("[*] WlanCloseHandle: Handle=0x%llx\n", hClientHandle);
}
