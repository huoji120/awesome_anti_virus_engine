#include "sandbox.h"
#include "sandbox_callbacks.h"
#include <fwpmu.h>

// FwpmEngineOpen0 API模拟
auto Api_FwpmEngineOpen0(void* sandbox, uc_engine* uc, uint64_t address)
    -> void {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t serverName = 0;
    uint64_t authnService = 0;
    uint64_t authIdentity = 0;
    uint64_t session = 0;
    uint64_t engineHandle = 0;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        uc_reg_read(uc, UC_X86_REG_RCX, &serverName);
        uc_reg_read(uc, UC_X86_REG_RDX, &authnService);
        uc_reg_read(uc, UC_X86_REG_R8, &authIdentity);
        uc_reg_read(uc, UC_X86_REG_R9, &session);
        uint64_t rsp;
        uc_reg_read(uc, UC_X86_REG_RSP, &rsp);
        uc_mem_read(uc, rsp + 0x28, &engineHandle, sizeof(engineHandle));
    } else {
        uint32_t esp;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp);
        esp += 4;  // 跳过返回地址
        uint32_t temp;
        uc_mem_read(uc, esp, &temp, sizeof(temp));
        serverName = temp;
        esp += 4;
        uc_mem_read(uc, esp, &temp, sizeof(temp));
        authnService = temp;
        esp += 4;
        uc_mem_read(uc, esp, &temp, sizeof(temp));
        authIdentity = temp;
        esp += 4;
        uc_mem_read(uc, esp, &temp, sizeof(temp));
        session = temp;
        esp += 4;
        uc_mem_read(uc, esp, &temp, sizeof(temp));
        engineHandle = temp;
    }

    // 创建新的WFP引擎实例
    auto engine = new Sandbox::FakeWFPEngine();
    engine->isOpen = true;
    HANDLE handle = context->GetNextWfpEngineHandle();
    context->GetWfpEngines()[handle] = engine;

    // 写回引擎句柄
    if (context->GetPeInfo()->isX64) {
        uc_mem_write(uc, engineHandle, &handle, sizeof(handle));
    } else {
        uint32_t handle32 = (uint32_t)(uint64_t)handle;
        uc_mem_write(uc, engineHandle, &handle32, sizeof(handle32));
    }

    // 返回成功
    uint64_t result = ERROR_SUCCESS;
    uc_reg_write(uc,
                 context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
                 &result);

    printf("[*] FwpmEngineOpen0: Handle=0x%llx\n", (uint64_t)handle);
}

// FwpmProviderAdd0 API模拟
auto Api_FwpmProviderAdd0(void* sandbox, uc_engine* uc, uint64_t address)
    -> void {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t engineHandle = 0;
    uint64_t provider = 0;
    uint64_t sd = 0;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        uc_reg_read(uc, UC_X86_REG_RCX, &engineHandle);
        uc_reg_read(uc, UC_X86_REG_RDX, &provider);
        uc_reg_read(uc, UC_X86_REG_R8, &sd);
    } else {
        uint32_t esp;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp);
        esp += 4;
        uint32_t temp;
        uc_mem_read(uc, esp, &temp, sizeof(temp));
        engineHandle = temp;
        esp += 4;
        uc_mem_read(uc, esp, &temp, sizeof(temp));
        provider = temp;
        esp += 4;
        uc_mem_read(uc, esp, &temp, sizeof(temp));
        sd = temp;
    }

    // 检查引擎句柄是否有效
    auto& engines = context->GetWfpEngines();
    auto it = engines.find((HANDLE)engineHandle);
    if (it == engines.end()) {
        uint64_t result = ERROR_INVALID_HANDLE;
        uc_reg_write(
            uc, context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
            &result);
        return;
    }

    // 添加提供者
    FWPM_PROVIDER0 providerData;
    uc_mem_read(uc, provider, &providerData, sizeof(FWPM_PROVIDER0));
    it->second->providers.push_back(providerData);

    // 返回成功
    uint64_t result = ERROR_SUCCESS;
    uc_reg_write(uc,
                 context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
                 &result);

    printf("[*] FwpmProviderAdd0: Handle=0x%llx\n", engineHandle);
}

// FwpmFilterAdd0 API模拟
auto Api_FwpmFilterAdd0(void* sandbox, uc_engine* uc, uint64_t address)
    -> void {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t engineHandle = 0;
    uint64_t filter = 0;
    uint64_t sd = 0;
    uint64_t id = 0;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        uc_reg_read(uc, UC_X86_REG_RCX, &engineHandle);
        uc_reg_read(uc, UC_X86_REG_RDX, &filter);
        uc_reg_read(uc, UC_X86_REG_R8, &sd);
        uc_reg_read(uc, UC_X86_REG_R9, &id);
    } else {
        uint32_t esp;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp);
        esp += 4;
        uint32_t temp;
        uc_mem_read(uc, esp, &temp, sizeof(temp));
        engineHandle = temp;
        esp += 4;
        uc_mem_read(uc, esp, &temp, sizeof(temp));
        filter = temp;
        esp += 4;
        uc_mem_read(uc, esp, &temp, sizeof(temp));
        sd = temp;
        esp += 4;
        uc_mem_read(uc, esp, &temp, sizeof(temp));
        id = temp;
    }

    // 检查引擎句柄是否有效
    auto& engines = context->GetWfpEngines();
    auto it = engines.find((HANDLE)engineHandle);
    if (it == engines.end()) {
        uint64_t result = ERROR_INVALID_HANDLE;
        uc_reg_write(
            uc, context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
            &result);
        return;
    }

    // 添加过滤器
    FWPM_FILTER0 filterData;
    uc_mem_read(uc, filter, &filterData, sizeof(FWPM_FILTER0));
    it->second->filters.push_back(filterData);

    // 生成并写回过滤器ID
    static uint64_t nextFilterId = 1;
    uint64_t filterId = nextFilterId++;
    if (id != 0) {
        uc_mem_write(uc, id, &filterId, sizeof(filterId));
    }

    // 返回成功
    uint64_t result = ERROR_SUCCESS;
    uc_reg_write(uc,
                 context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
                 &result);

    printf("[*] FwpmFilterAdd0: Handle=0x%llx, FilterId=0x%llx\n", engineHandle,
           filterId);
}

// FwpmEngineClose0 API模拟
auto Api_FwpmEngineClose0(void* sandbox, uc_engine* uc, uint64_t address)
    -> void {
    auto context = static_cast<Sandbox*>(sandbox);
    uint64_t engineHandle = 0;

    // 获取参数
    if (context->GetPeInfo()->isX64) {
        uc_reg_read(uc, UC_X86_REG_RCX, &engineHandle);
    } else {
        uint32_t esp;
        uc_reg_read(uc, UC_X86_REG_ESP, &esp);
        esp += 4;
        uint32_t temp;
        uc_mem_read(uc, esp, &temp, sizeof(temp));
        engineHandle = temp;
    }

    // 检查并关闭引擎
    auto& engines = context->GetWfpEngines();
    auto it = engines.find((HANDLE)engineHandle);
    if (it != engines.end()) {
        delete it->second;
        engines.erase(it);
    }

    // 返回成功
    uint64_t result = ERROR_SUCCESS;
    uc_reg_write(uc,
                 context->GetPeInfo()->isX64 ? UC_X86_REG_RAX : UC_X86_REG_EAX,
                 &result);

    printf("[*] FwpmEngineClose0: Handle=0x%llx\n", engineHandle);
}