# gpu_scheduler.py
#
# Simple Virtual GPU Scheduler
#
# Purpose:
#   - Simulate GPU threads, warps and thread blocks
#   - Execute the same instruction across threads in a warp
#   - Share GPUMemory between all threads
#   - Give each thread its own register file
#
# Dependencies:
#   gpu_memory.py
#   gpu_isa.py
#   gpu_compute.py
#
# Execution hierarchy:
#
#   GPUScheduler
#        |
#        +-- ThreadBlock
#               |
#               +-- Warp
#                      |
#                      +-- GPUThread
#                      +-- GPUThread
#                      +-- GPUThread
#
# This is an educational GPU scheduler.
# It does not yet simulate actual parallel execution.


from dataclasses import dataclass, field
from typing import List

from gpu_memory import GPUMemory

from gpu_isa import (
    Opcode,
    Instruction,
    GPUProgram,
)

from gpu_compute import GPUComputeCore


@dataclass
class GPUThread:
    """One virtual GPU thread."""

    thread_id: int
    block_id: int
    warp_id: int
    core: GPUComputeCore
    pc: int = 0
    active: bool = True
    halted: bool = False

    def reset(self) -> None:
        self.pc = 0
        self.active = True
        self.halted = False
        self.core.reset()

    def execute(self, instruction: Instruction) -> None:
        if not self.active or self.halted:
            return

        self.core.execute_instruction(instruction)

        if instruction.opcode == Opcode.HALT:
            self.halted = True
            self.active = False
            return

        self.pc += 1


@dataclass
class Warp:
    """Virtual GPU warp executing one instruction across several threads."""

    warp_id: int
    block_id: int
    threads: List[GPUThread] = field(default_factory=list)

    def is_finished(self) -> bool:
        return all(thread.halted for thread in self.threads)

    def active_thread_count(self) -> int:
        return sum(1 for thread in self.threads if thread.active)

    def execute_instruction(self, instruction: Instruction) -> None:
        for thread in self.threads:
            if thread.active:
                thread.execute(instruction)


@dataclass
class ThreadBlock:
    """Virtual GPU thread block containing one or more warps."""

    block_id: int
    warps: List[Warp] = field(default_factory=list)

    def is_finished(self) -> bool:
        return all(warp.is_finished() for warp in self.warps)

    def thread_count(self) -> int:
        return sum(len(warp.threads) for warp in self.warps)


class GPUScheduler:
    """Simple virtual GPU round-robin warp scheduler."""

    def __init__(
        self,
        memory: GPUMemory,
        warp_size: int = 4,
        num_registers: int = 32
    ):
        if warp_size <= 0:
            raise ValueError("warp_size must be greater than 0.")
        if num_registers <= 0:
            raise ValueError("num_registers must be greater than 0.")

        self.memory = memory
        self.warp_size = warp_size
        self.num_registers = num_registers
        self.blocks: List[ThreadBlock] = []
        self.total_threads = 0
        self.cycle_count = 0
        self.instruction_count = 0

    def create_grid(self, num_blocks: int, threads_per_block: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be greater than 0.")
        if threads_per_block <= 0:
            raise ValueError("threads_per_block must be greater than 0.")

        self.blocks.clear()
        global_thread_id = 0
        global_warp_id = 0

        for block_id in range(num_blocks):
            block = ThreadBlock(block_id=block_id)
            remaining_threads = threads_per_block
            local_thread_id = 0

            while remaining_threads > 0:
                warp = Warp(
                    warp_id=global_warp_id,
                    block_id=block_id
                )

                threads_in_warp = min(self.warp_size, remaining_threads)

                for _ in range(threads_in_warp):
                    core = GPUComputeCore(
                        memory=self.memory,
                        num_registers=self.num_registers
                    )

                    thread = GPUThread(
                        thread_id=global_thread_id,
                        block_id=block_id,
                        warp_id=global_warp_id,
                        core=core
                    )

                    self._initialize_thread_registers(
                        thread=thread,
                        global_thread_id=global_thread_id,
                        local_thread_id=local_thread_id,
                        warp_id=global_warp_id,
                        block_id=block_id
                    )

                    warp.threads.append(thread)
                    global_thread_id += 1
                    local_thread_id += 1
                    remaining_threads -= 1

                block.warps.append(warp)
                global_warp_id += 1

            self.blocks.append(block)

        self.total_threads = global_thread_id

    def _initialize_thread_registers(
        self,
        thread: GPUThread,
        global_thread_id: int,
        local_thread_id: int,
        warp_id: int,
        block_id: int
    ) -> None:
        special_registers = {
            28: global_thread_id,
            29: local_thread_id,
            30: warp_id,
            31: block_id,
        }

        for register, value in special_registers.items():
            if register < self.num_registers:
                thread.core.write_register(register, float(value))

    def run(self, program: GPUProgram, max_cycles: int = 1_000_000) -> None:
        if not self.blocks:
            raise RuntimeError(
                "GPU grid has not been created. Call create_grid() first."
            )

        if len(program) == 0:
            return

        self.cycle_count = 0
        self.instruction_count = 0

        for block in self.blocks:
            for warp in block.warps:
                for thread in warp.threads:
                    thread.pc = 0
                    thread.active = True
                    thread.halted = False

        while not self.is_finished():
            if self.cycle_count >= max_cycles:
                raise RuntimeError("Maximum GPU scheduler cycles exceeded.")

            progress = False

            for block in self.blocks:
                for warp in block.warps:
                    if warp.is_finished():
                        continue

                    self._execute_warp(warp, program)
                    progress = True

            if not progress:
                break

            self.cycle_count += 1

    def _execute_warp(self, warp: Warp, program: GPUProgram) -> None:
        active_threads = [
            thread
            for thread in warp.threads
            if thread.active and not thread.halted
        ]

        if not active_threads:
            return

        pc = active_threads[0].pc

        if pc >= len(program):
            for thread in active_threads:
                thread.active = False
                thread.halted = True
            return

        instruction = program[pc]

        for thread in active_threads:
            if thread.pc != pc:
                raise RuntimeError(
                    "Warp divergence detected. Branch handling is not implemented yet."
                )

            thread.execute(instruction)
            self.instruction_count += 1

    def is_finished(self) -> bool:
        return all(block.is_finished() for block in self.blocks)

    def dump_grid(self) -> None:
        print("GPU Grid")
        print("========")

        for block in self.blocks:
            print(f"Block {block.block_id}")

            for warp in block.warps:
                thread_ids = [thread.thread_id for thread in warp.threads]
                print(f"  Warp {warp.warp_id}: Threads {thread_ids}")

    def dump_threads(self, register_count: int = 4) -> None:
        print("GPU Thread State")
        print("================")

        for block in self.blocks:
            for warp in block.warps:
                for thread in warp.threads:
                    count = min(register_count, self.num_registers)
                    registers = [
                        thread.core.read_register(i)
                        for i in range(count)
                    ]

                    print(
                        f"Thread {thread.thread_id:03d} "
                        f"Block={thread.block_id} "
                        f"Warp={thread.warp_id} "
                        f"PC={thread.pc} "
                        f"Registers={registers}"
                    )

    def info(self) -> None:
        total_warps = sum(len(block.warps) for block in self.blocks)

        print("GPU Scheduler Information")
        print("=========================")
        print(f"Blocks       : {len(self.blocks)}")
        print(f"Warps        : {total_warps}")
        print(f"Threads      : {self.total_threads}")
        print(f"Warp size    : {self.warp_size}")
        print(f"Cycles       : {self.cycle_count}")
        print(f"Instructions : {self.instruction_count}")


if __name__ == "__main__":
    from gpu_isa import reg, imm, Instruction, Opcode, GPUProgram

    print()
    print("================================")
    print("Virtual GPU Scheduler Test")
    print("================================")
    print()

    memory = GPUMemory(size=1024)

    scheduler = GPUScheduler(
        memory=memory,
        warp_size=4,
        num_registers=32
    )

    scheduler.create_grid(
        num_blocks=2,
        threads_per_block=8
    )

    scheduler.dump_grid()
    print()

    program = GPUProgram()

    program.add(
        Instruction(
            opcode=Opcode.MOV,
            dst=reg(0),
            src1=imm(10.0)
        )
    )

    program.add(
        Instruction(
            opcode=Opcode.MOV,
            dst=reg(1),
            src1=imm(20.0)
        )
    )

    program.add(
        Instruction(
            opcode=Opcode.ADD,
            dst=reg(2),
            src1=reg(0),
            src2=reg(1)
        )
    )

    program.add(
        Instruction(opcode=Opcode.HALT)
    )

    scheduler.run(program)

    print()
    scheduler.dump_threads(register_count=4)
    print()
    scheduler.info()
    print()

    print("Special Registers")
    print("=================")

    for block in scheduler.blocks:
        for warp in block.warps:
            for thread in warp.threads:
                print(
                    f"Thread {thread.thread_id:02d}: "
                    f"R28(global_tid)={thread.core.read_register(28):.0f}, "
                    f"R29(local_tid)={thread.core.read_register(29):.0f}, "
                    f"R30(warp_id)={thread.core.read_register(30):.0f}, "
                    f"R31(block_id)={thread.core.read_register(31):.0f}"
                )
