#include "sandbox.h"
#include "sandbox_callbacks.h"
#include "sandbox_api_winhttp.h"
#include <tlhelp32.h>
auto Api_ReadFile(void* sandbox, uc_engine* uc, uint64_t address) -> void {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t hFile = 0;
    uint64_t lpBuffer = 0;
    uint32_t nNumberOfBytesToRead = 0;
    uint64_t lpNumberOfBytesRead = 0;
    uint64_t lpOverlapped = 0;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        // x64: rcx = hFile, rdx = lpBuffer, r8 = nNumberOfBytesToRead, r9 =
        // lpNumberOfBytesRead
        uc_reg_read(uc, UC_X86_REG_RCX, &hFile);
        uc_reg_read(uc, UC_X86_REG_RDX, &lpBuffer);
        uint64_t temp_bytes_to_read;
        uc_reg_read(uc, UC_X86_REG_R8, &temp_bytes_to_read);
        nNumberOfBytesToRead = static_cast<uint32_t>(temp_bytes_to_read);
        uc_reg_read(uc, UC_X86_REG_R9, &lpNumberOfBytesRead);

        // 从栈上读取lpOverlapped参数
        uint64_t rsp;
        uc_reg_read(uc, UC_X86_REG_RSP, &rsp);
        uc_mem_read(uc, rsp + 0x28, &lpOverlapped, sizeof(uint64_t));
    } else {
        // x86: 从栈上读取参数
        uint32_t esp;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp);
        esp += 0x4;  // 跳过返回地址

        uint32_t temp_handle, temp_buffer, temp_bytes_read, temp_overlapped;
        uc_mem_read(uc, esp, &temp_handle, sizeof(uint32_t));
        uc_mem_read(uc, esp + 0x4, &temp_buffer, sizeof(uint32_t));
        uc_mem_read(uc, esp + 0x8, &nNumberOfBytesToRead, sizeof(uint32_t));
        uc_mem_read(uc, esp + 0xC, &temp_bytes_read, sizeof(uint32_t));
        uc_mem_read(uc, esp + 0x10, &temp_overlapped, sizeof(uint32_t));

        hFile = temp_handle;
        lpBuffer = temp_buffer;
        lpNumberOfBytesRead = temp_bytes_read;
        lpOverlapped = temp_overlapped;
    }

    // 检查句柄是否为之前CreatePipe创建的读管道
    bool success = false;
    uint32_t bytesRead = 0;

    // 检查是否为之前CreatePipe创建的读管道句柄 (0x1338)
    if (hFile == 0x1338) {
        // 模拟管道数据读取 - 这里我们可以根据之前的CreateProcessA来生成数据
        // 生成一些模拟的ping命令输出数据
        std::string pingOutput =
            "Reply from 127.0.0.1: bytes=32 time<1ms TTL=45\r\n"
            "Reply from 127.0.0.1: bytes=32 time<1ms TTL=45\r\n"
            "Reply from 127.0.0.1: bytes=32 time<1ms TTL=45\r\n"
            "Reply from 127.0.0.1: bytes=32 time<1ms TTL=45\r\n\r\n"
            "Ping statistics for 127.0.0.1:\r\n"
            "    Packets: Sent = 4, Received = 4, Lost = 0 (0% loss),\r\n"
            "Approximate round trip times in milli-seconds:\r\n"
            "    Minimum = 0ms, Maximum = 0ms, Average = 0ms\r\n";

        // 确保不超过缓冲区大小
        bytesRead = min(static_cast<uint32_t>(pingOutput.length()),
                        nNumberOfBytesToRead);

        // 写入数据到缓冲区
        if (lpBuffer != 0 && bytesRead > 0) {
            uc_mem_write(uc, lpBuffer, pingOutput.c_str(), bytesRead);
            success = true;
        }
    }

    // 写入实际读取的字节数
    if (lpNumberOfBytesRead != 0) {
        if (context->GetPeInfo()->isX64) {
            uc_mem_write(uc, lpNumberOfBytesRead, &bytesRead, sizeof(uint32_t));
        } else {
            uc_mem_write(uc, lpNumberOfBytesRead, &bytesRead, sizeof(uint32_t));
        }
    }

    printf(
        "[*] ReadFile: Handle=0x%llx, Buffer=0x%llx, BytesToRead=%u, "
        "BytesRead=%u, Success=%d\n",
        hFile, lpBuffer, nNumberOfBytesToRead, bytesRead, success);

    // 设置返回值
    uint64_t result = success ? 1 : 0;
    uc_reg_write(uc,
                 context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
                 &result);

    // 如果失败，设置错误码
    if (!success) {
        DWORD error = ERROR_INVALID_HANDLE;
        if (context->GetPeInfo()->isX64) {
            context->GetTeb64()->LastErrorValue = error;
        } else {
            context->GetTeb32()->LastErrorValue = error;
        }
    }
}
auto Api_CreatePipe(void* sandbox, uc_engine* uc, uint64_t address) -> void {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t hReadPipe = 0;
    uint64_t hWritePipe = 0;
    uint64_t lpPipeAttributes = 0;
    uint32_t nSize = 0;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        // x64: rcx = hReadPipe, rdx = hWritePipe, r8 = lpPipeAttributes, r9 =
        // nSize
        uc_reg_read(uc, UC_X86_REG_RCX, &hReadPipe);
        uc_reg_read(uc, UC_X86_REG_RDX, &hWritePipe);
        uc_reg_read(uc, UC_X86_REG_R8, &lpPipeAttributes);
        uint64_t temp_size;
        uc_reg_read(uc, UC_X86_REG_R9, &temp_size);
        nSize = static_cast<uint32_t>(temp_size);
    } else {
        // x86: 从栈上读取参数
        uint32_t esp_address = 0;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp_address);
        esp_address += 0x4;  // 跳过返回地址

        uint32_t temp_read_pipe;
        uint32_t temp_write_pipe;
        uint32_t temp_pipe_attributes;

        uc_mem_read(uc, esp_address, &temp_read_pipe, sizeof(uint32_t));
        uc_mem_read(uc, esp_address + 0x4, &temp_write_pipe, sizeof(uint32_t));
        uc_mem_read(uc, esp_address + 0x8, &temp_pipe_attributes,
                    sizeof(uint32_t));
        uc_mem_read(uc, esp_address + 0xC, &nSize, sizeof(uint32_t));

        hReadPipe = temp_read_pipe;
        hWritePipe = temp_write_pipe;
        lpPipeAttributes = temp_pipe_attributes;
    }

    // 创建模拟的管道句柄
    uint64_t read_handle = 0x1338;   // 使用特殊值作为读取句柄
    uint64_t write_handle = 0x1339;  // 使用特殊值作为写入句柄

    // 生成唯一的管道名称
    char pipeName[MAX_PATH];
    DWORD processId =
        context->GetPeInfo()->isX64
            ? static_cast<DWORD>(context->GetTeb64()->ClientId.UniqueProcess)
            : static_cast<DWORD>(context->GetTeb32()->ClientId.UniqueProcess);

    snprintf(pipeName, sizeof(pipeName),
             "\\\\.\\pipe\\sandbox_pipe_%lu_%llx_%llx", processId, read_handle,
             write_handle);

    // 写入句柄到输出参数
    if (context->GetPeInfo()->isX64) {
        uc_mem_write(uc, hReadPipe, &read_handle, sizeof(uint64_t));
        uc_mem_write(uc, hWritePipe, &write_handle, sizeof(uint64_t));
    } else {
        uint32_t read_handle_32 = static_cast<uint32_t>(read_handle);
        uint32_t write_handle_32 = static_cast<uint32_t>(write_handle);
        uc_mem_write(uc, hReadPipe, &read_handle_32, sizeof(uint32_t));
        uc_mem_write(uc, hWritePipe, &write_handle_32, sizeof(uint32_t));
    }

    // 设置返回值为TRUE
    uint64_t result = 1;
    uc_reg_write(uc,
                 context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
                 &result);

    printf(
        "[*] CreatePipe: Name=%s, ReadHandle=0x%llx, WriteHandle=0x%llx, "
        "Size=%u\n",
        pipeName, read_handle, write_handle, nSize);
}