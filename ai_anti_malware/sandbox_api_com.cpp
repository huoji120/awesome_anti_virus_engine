#include "sandbox.h"
#include "sandbox_callbacks.h"
#include "sandbox_api_winhttp.h"
#include <tlhelp32.h>
#include "sandbox_api_com.h"


// 自定义函数：将GUID转换为字符串
void GUIDToString(const GUID& guid, char* str, size_t size) {
    // GUID格式: {XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}
    snprintf(str, size, "{%08X-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X}",
             guid.Data1, guid.Data2, guid.Data3, guid.Data4[0], guid.Data4[1],
             guid.Data4[2], guid.Data4[3], guid.Data4[4], guid.Data4[5],
             guid.Data4[6], guid.Data4[7]);
}

// COM API 实现
void Api_CoInitializeEx(void* sandbox, uc_engine* uc, uint64_t address) {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t pvReserved = 0;
    uint32_t dwCoInit = 0;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        // x64: rcx = pvReserved, rdx = dwCoInit
        uc_reg_read(uc, UC_X86_REG_RCX, &pvReserved);
        uc_reg_read(uc, UC_X86_REG_RDX, &dwCoInit);
    } else {
        // x86: 从栈上读取参数
        uint32_t esp_address = 0;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp_address);
        esp_address += 0x4;  // 跳过返回地址

        uint32_t temp_reserved;
        uint32_t temp_coinit;
        uc_mem_read(uc, esp_address, &temp_reserved, sizeof(uint32_t));
        uc_mem_read(uc, esp_address + 0x4, &temp_coinit, sizeof(uint32_t));
        pvReserved = temp_reserved;
        dwCoInit = temp_coinit;
    }

    // 打印日志
    printf("[*] CoInitializeEx: pvReserved=0x%llx, dwCoInit=0x%x\n", pvReserved,
           dwCoInit);

    // 返回S_OK (0)
    uint32_t result = 0;  // S_OK = 0
    uc_reg_write(uc,
                 context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
                 &result);
}

void Api_CoCreateInstance(void* sandbox, uc_engine* uc, uint64_t address) {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t rclsid = 0;        // REFCLSID
    uint64_t pUnkOuter = 0;     // LPUNKNOWN
    uint32_t dwClsContext = 0;  // DWORD
    uint64_t riid = 0;          // REFIID
    uint64_t ppv = 0;           // LPVOID*

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        // x64: rcx = rclsid, rdx = pUnkOuter, r8 = dwClsContext, r9 = riid
        uc_reg_read(uc, UC_X86_REG_RCX, &rclsid);
        uc_reg_read(uc, UC_X86_REG_RDX, &pUnkOuter);
        uint64_t temp_context;
        uc_reg_read(uc, UC_X86_REG_R8, &temp_context);
        dwClsContext = static_cast<uint32_t>(temp_context);
        uc_reg_read(uc, UC_X86_REG_R9, &riid);

        // 从栈上读取最后一个参数
        uint64_t rsp;
        uc_reg_read(uc, UC_X86_REG_RSP, &rsp);
        uc_mem_read(uc, rsp + 0x28, &ppv, sizeof(uint64_t));
    } else {
        // x86: 从栈上读取参数
        uint32_t esp_address = 0;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp_address);
        esp_address += 0x4;  // 跳过返回地址

        uint32_t temp_params[5];  // 所有参数都是32位的
        for (int i = 0; i < 5; i++) {
            uc_mem_read(uc, esp_address + (i * 4), &temp_params[i],
                        sizeof(uint32_t));
        }

        rclsid = temp_params[0];
        pUnkOuter = temp_params[1];
        dwClsContext = temp_params[2];
        riid = temp_params[3];
        ppv = temp_params[4];
    }

    // 读取并打印CLSID
    CLSID clsid;
    if (rclsid != 0) {
        uc_mem_read(uc, rclsid, &clsid, sizeof(CLSID));

        // 将CLSID转换为字符串并打印
        char clsidStr[40] = {0};
        GUIDToString(clsid, clsidStr, sizeof(clsidStr));

        printf("[*] CoCreateInstance: CLSID=%s, Context=0x%x\n", clsidStr,
               dwClsContext);

        // 也打印IID (接口ID)
        if (riid != 0) {
            IID iid;
            uc_mem_read(uc, riid, &iid, sizeof(IID));
            char iidStr[40] = {0};
            GUIDToString(iid, iidStr, sizeof(iidStr));
            printf("[*] CoCreateInstance: IID=%s\n", iidStr);
        }
    } else {
        printf("[*] CoCreateInstance: CLSID=NULL, Context=0x%x\n",
               dwClsContext);
    }

    // 创建一个假的接口指针
    uint64_t fake_interface = 0xABABABABAB;


    // 如果ppv有效，将fake_interface写入ppv指向的位置
    if (ppv != 0) {
        if (context->GetPeInfo()->isX64) {
            uc_mem_write(uc, ppv, &fake_interface, sizeof(uint64_t));
        } else {
            uint32_t interface_32 = static_cast<uint32_t>(fake_interface);
            uc_mem_write(uc, (uint32_t)ppv, &interface_32, sizeof(uint32_t));
        }
    }

    // 返回S_OK (0)
    uint32_t result = 0;  // S_OK = 0
    uc_reg_write(uc,
                 context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
                 &result);
}

void Api_VariantInit(void* sandbox, uc_engine* uc, uint64_t address) {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t pvarg = 0;  // 指向VARIANT的指针

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        // x64: rcx = pvarg
        uc_reg_read(uc, UC_X86_REG_RCX, &pvarg);
    } else {
        // x86: 从栈上读取参数
        uint32_t esp_address = 0;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp_address);
        esp_address += 0x4;  // 跳过返回地址

        uint32_t temp_pvarg;
        uc_mem_read(uc, esp_address, &temp_pvarg, sizeof(uint32_t));
        pvarg = temp_pvarg;
    }

    // 检查pvarg是否有效
    if (pvarg != 0) {
        // 初始化VARIANT结构体为VT_EMPTY
        VARTYPE vt = VT_EMPTY;
        uc_mem_write(uc, pvarg, &vt, sizeof(VARTYPE));

        // 清零VARIANT结构体的其余部分
        uint8_t zeros[14] = {0};  // VARIANT结构体大小为16字节，前2字节为vt
        uc_mem_write(uc, pvarg + sizeof(VARTYPE), zeros, sizeof(zeros));
    }

    printf("[*] VariantInit: pvarg=0x%llx\n", pvarg);
}

void Api_VariantClear(void* sandbox, uc_engine* uc, uint64_t address) {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t pvarg = 0;  // 指向VARIANT的指针

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        // x64: rcx = pvarg
        uc_reg_read(uc, UC_X86_REG_RCX, &pvarg);
    } else {
        // x86: 从栈上读取参数
        uint32_t esp_address = 0;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp_address);
        esp_address += 0x4;  // 跳过返回地址

        uint32_t temp_pvarg;
        uc_mem_read(uc, esp_address, &temp_pvarg, sizeof(uint32_t));
        pvarg = temp_pvarg;
    }

    // 检查pvarg是否有效
    if (pvarg != 0) {
        // 读取当前的VARIANT类型
        VARTYPE vt;
        uc_mem_read(uc, pvarg, &vt, sizeof(VARTYPE));

        // 将类型重置为VT_EMPTY并清零其余部分
        vt = VT_EMPTY;
        uc_mem_write(uc, pvarg, &vt, sizeof(VARTYPE));

        uint8_t zeros[14] = {0};
        uc_mem_write(uc, pvarg + sizeof(VARTYPE), zeros, sizeof(zeros));
    }

    // 返回S_OK
    uint32_t result = 0;  // S_OK
    uc_reg_write(uc,
                 context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
                 &result);

    printf("[*] VariantClear: pvarg=0x%llx\n", pvarg);
}

void Api_SysAllocString(void* sandbox, uc_engine* uc, uint64_t address) {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t psz = 0;  // 源字符串指针

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        // x64: rcx = psz
        uc_reg_read(uc, UC_X86_REG_RCX, &psz);
    } else {
        // x86: 从栈上读取参数
        uint32_t esp_address = 0;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp_address);
        esp_address += 0x4;  // 跳过返回地址

        uint32_t temp_psz;
        uc_mem_read(uc, esp_address, &temp_psz, sizeof(uint32_t));
        psz = temp_psz;
    }

    uint64_t bstr_ptr = 0;

    // 如果源字符串有效
    if (psz != 0) {
        // 计算源字符串长度
        size_t len = 0;
        wchar_t wch;
        do {
            uc_mem_read(uc, psz + (len * sizeof(wchar_t)), &wch,
                        sizeof(wchar_t));
            len++;
        } while (wch != 0 && len < 1024);  // 设置一个合理的最大长度限制
        len--;  // 不包括null终止符

        // 为BSTR分配内存：4字节长度 + 字符串内容 + 终止符
        size_t bstr_size = sizeof(uint32_t) + (len + 1) * sizeof(wchar_t);
        bstr_ptr = context->AllocateMemory(bstr_size);

        if (bstr_ptr != 0) {
            // 写入字符串长度（字节数）
            uint32_t byte_len = static_cast<uint32_t>(len * sizeof(wchar_t));
            uc_mem_write(uc, bstr_ptr, &byte_len, sizeof(uint32_t));

            // 写入字符串内容
            uint64_t string_offset = bstr_ptr + sizeof(uint32_t);
            uc_mem_write(uc, string_offset, (void*)psz, byte_len);

            // 添加终止符
            wchar_t null_char = 0;
            uc_mem_write(uc, string_offset + byte_len, &null_char,
                         sizeof(wchar_t));

            // BSTR指针指向字符串内容，不包括长度前缀
            bstr_ptr = string_offset;
        }
    }

    // 返回BSTR指针
    uc_reg_write(uc,
                 context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
                 &bstr_ptr);

    printf("[*] SysAllocString: psz=0x%llx, result=0x%llx\n", psz, bstr_ptr);
}
