# gpu_compute.py
#
# Simple Virtual GPU Compute Core
#
# Purpose:
#   - Execute instructions defined in gpu_isa.py
#   - Use memory provided by gpu_memory.py
#   - Provide a simple register file
#   - Execute scalar arithmetic, FMA, MATMUL and SOFTMAX
#
# Dependencies:
#   gpu_memory.py
#   gpu_isa.py

import math
from typing import List

from gpu_memory import GPUMemory
from gpu_isa import (
    Opcode,
    Operand,
    OperandType,
    Instruction,
    GPUProgram,
    validate_instruction,
)


class GPUComputeCore:
    """
    Simple virtual GPU compute core.

    This version simulates:
      - register file
      - instruction execution
      - memory access
      - scalar arithmetic
      - matrix multiplication
      - softmax

    It does not yet simulate:
      - warp scheduling
      - thread blocks
      - shared memory
      - SIMD lanes
    """

    def __init__(
        self,
        memory: GPUMemory,
        num_registers: int = 32
    ):
        if num_registers <= 0:
            raise ValueError(
                "num_registers must be greater than 0."
            )

        self.memory = memory
        self.num_registers = num_registers

        self.registers: List[float] = [
            0.0
        ] * num_registers

        self.pc = 0
        self.halted = False

    # ========================================================
    # Register Access
    # ========================================================

    def read_register(self, index: int) -> float:
        self._check_register(index)
        return self.registers[index]

    def write_register(
        self,
        index: int,
        value: float
    ) -> None:
        self._check_register(index)
        self.registers[index] = float(value)

    def _check_register(self, index: int) -> None:
        if index < 0 or index >= self.num_registers:
            raise IndexError(
                f"Invalid register index: R{index}"
            )

    # ========================================================
    # Operand Access
    # ========================================================

    def read_operand(
        self,
        operand: Operand
    ) -> float:

        if operand.type == OperandType.REGISTER:
            return self.read_register(
                int(operand.value)
            )

        elif operand.type == OperandType.MEMORY:
            return self.memory.read_scalar(
                int(operand.value)
            )

        elif operand.type == OperandType.IMMEDIATE:
            return float(operand.value)

        raise ValueError(
            f"Unsupported operand type: {operand.type}"
        )

    def write_operand(
        self,
        operand: Operand,
        value: float
    ) -> None:

        if operand.type == OperandType.REGISTER:
            self.write_register(
                int(operand.value),
                value
            )
            return

        elif operand.type == OperandType.MEMORY:
            self.memory.write_scalar(
                int(operand.value),
                value
            )
            return

        raise ValueError(
            "Destination operand must be "
            "REGISTER or MEMORY."
        )

    # ========================================================
    # Instruction Execution
    # ========================================================

    def execute_instruction(
        self,
        inst: Instruction
    ) -> None:

        validate_instruction(inst)

        opcode = inst.opcode

        if opcode == Opcode.NOP:
            return

        if opcode == Opcode.HALT:
            self.halted = True
            return

        if opcode == Opcode.LOAD:
            value = self.memory.read_scalar(
                int(inst.src1.value)
            )

            self.write_register(
                int(inst.dst.value),
                value
            )

            return

        if opcode == Opcode.STORE:
            value = self.read_register(
                int(inst.src1.value)
            )

            self.memory.write_scalar(
                int(inst.dst.value),
                value
            )

            return

        if opcode == Opcode.MOV:
            value = self.read_operand(
                inst.src1
            )

            self.write_register(
                int(inst.dst.value),
                value
            )

            return

        if opcode == Opcode.ADD:
            a = self.read_operand(inst.src1)
            b = self.read_operand(inst.src2)

            self.write_register(
                int(inst.dst.value),
                a + b
            )

            return

        if opcode == Opcode.SUB:
            a = self.read_operand(inst.src1)
            b = self.read_operand(inst.src2)

            self.write_register(
                int(inst.dst.value),
                a - b
            )

            return

        if opcode == Opcode.MUL:
            a = self.read_operand(inst.src1)
            b = self.read_operand(inst.src2)

            self.write_register(
                int(inst.dst.value),
                a * b
            )

            return

        if opcode == Opcode.DIV:
            a = self.read_operand(inst.src1)
            b = self.read_operand(inst.src2)

            if b == 0.0:
                raise ZeroDivisionError(
                    "GPU DIV by zero."
                )

            self.write_register(
                int(inst.dst.value),
                a / b
            )

            return

        if opcode == Opcode.FMA:
            a = self.read_operand(inst.src1)
            b = self.read_operand(inst.src2)
            c = self.read_operand(inst.src3)

            result = a * b + c

            self.write_register(
                int(inst.dst.value),
                result
            )

            return

        if opcode == Opcode.MATMUL:
            self._execute_matmul(inst)
            return

        if opcode == Opcode.SOFTMAX:
            self._execute_softmax(inst)
            return

        raise ValueError(
            f"Unsupported opcode: {opcode}"
        )

    # ========================================================
    # MATMUL
    # ========================================================

    def _execute_matmul(
        self,
        inst: Instruction
    ) -> None:

        if (
            inst.dst.type != OperandType.MEMORY
            or inst.src1.type != OperandType.MEMORY
            or inst.src2.type != OperandType.MEMORY
        ):
            raise ValueError(
                "MATMUL operands must use MEMORY."
            )

        m = int(inst.extra["m"])
        n = int(inst.extra["n"])
        k = int(inst.extra["k"])

        if m <= 0 or n <= 0 or k <= 0:
            raise ValueError(
                "MATMUL dimensions must be > 0."
            )

        addr_c = int(inst.dst.value)
        addr_a = int(inst.src1.value)
        addr_b = int(inst.src2.value)

        for row in range(m):
            for col in range(n):
                acc = 0.0

                for inner in range(k):
                    a_index = addr_a + row * k + inner
                    b_index = addr_b + inner * n + col

                    a = self.memory.read_scalar(a_index)
                    b = self.memory.read_scalar(b_index)

                    acc += a * b

                c_index = addr_c + row * n + col

                self.memory.write_scalar(
                    c_index,
                    acc
                )

    # ========================================================
    # SOFTMAX
    # ========================================================

    def _execute_softmax(
        self,
        inst: Instruction
    ) -> None:

        if (
            inst.dst.type != OperandType.MEMORY
            or inst.src1.type != OperandType.MEMORY
        ):
            raise ValueError(
                "SOFTMAX operands must use MEMORY."
            )

        size = inst.size

        src_address = int(inst.src1.value)
        dst_address = int(inst.dst.value)

        values = self.memory.read(
            src_address,
            size
        )

        max_value = max(values)

        exp_values = [
            math.exp(v - max_value)
            for v in values
        ]

        total = sum(exp_values)

        if total == 0.0:
            raise ZeroDivisionError(
                "SOFTMAX normalization total is zero."
            )

        result = [
            value / total
            for value in exp_values
        ]

        self.memory.write(
            dst_address,
            result
        )

    # ========================================================
    # Program Execution
    # ========================================================

    def run(
        self,
        program: GPUProgram,
        max_instructions: int = 1_000_000
    ) -> None:

        self.pc = 0
        self.halted = False

        executed = 0

        while (
            self.pc < len(program)
            and not self.halted
        ):

            if executed >= max_instructions:
                raise RuntimeError(
                    "Maximum instruction count exceeded."
                )

            inst = program[self.pc]

            self.execute_instruction(inst)

            self.pc += 1
            executed += 1

    # ========================================================
    # Debug
    # ========================================================

    def dump_registers(
        self,
        count: int = 8
    ) -> None:

        count = min(
            count,
            self.num_registers
        )

        print("GPU Register File")
        print("-----------------")

        for i in range(count):
            print(
                f"R{i:02d} = "
                f"{self.registers[i]}"
            )

    def reset(self) -> None:
        self.registers = [
            0.0
        ] * self.num_registers

        self.pc = 0
        self.halted = False


if __name__ == "__main__":

    from gpu_isa import reg, mem, imm

    print("================================")
    print("Simple Arithmetic Test")
    print("================================")

    memory = GPUMemory(size=256)
    core = GPUComputeCore(
        memory=memory,
        num_registers=16
    )

    addr_a = memory.allocate(1)
    addr_b = memory.allocate(1)
    addr_result = memory.allocate(1)

    memory.write_scalar(addr_a, 10.0)
    memory.write_scalar(addr_b, 20.0)

    program = GPUProgram()

    program.add(
        Instruction(
            opcode=Opcode.LOAD,
            dst=reg(0),
            src1=mem(addr_a)
        )
    )

    program.add(
        Instruction(
            opcode=Opcode.LOAD,
            dst=reg(1),
            src1=mem(addr_b)
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
        Instruction(
            opcode=Opcode.STORE,
            dst=mem(addr_result),
            src1=reg(2)
        )
    )

    program.add(
        Instruction(
            opcode=Opcode.HALT
        )
    )

    core.run(program)

    print(
        "10 + 20 =",
        memory.read_scalar(addr_result)
    )

    core.dump_registers(count=4)

    print()
    print("================================")
    print("FMA Test")
    print("================================")

    fma_program = GPUProgram()

    fma_program.add(
        Instruction(
            opcode=Opcode.MOV,
            dst=reg(0),
            src1=imm(2.0)
        )
    )

    fma_program.add(
        Instruction(
            opcode=Opcode.MOV,
            dst=reg(1),
            src1=imm(3.0)
        )
    )

    fma_program.add(
        Instruction(
            opcode=Opcode.MOV,
            dst=reg(2),
            src1=imm(4.0)
        )
    )

    fma_program.add(
        Instruction(
            opcode=Opcode.FMA,
            dst=reg(3),
            src1=reg(0),
            src2=reg(1),
            src3=reg(2)
        )
    )

    fma_program.add(
        Instruction(
            opcode=Opcode.HALT
        )
    )

    core.run(fma_program)

    print(
        "2 * 3 + 4 =",
        core.read_register(3)
    )

    print()
    print("================================")
    print("MATMUL Test")
    print("================================")

    addr_matrix_a = memory.allocate(4)
    addr_matrix_b = memory.allocate(4)
    addr_matrix_c = memory.allocate(4)

    memory.write(
        addr_matrix_a,
        [
            1.0, 2.0,
            3.0, 4.0
        ]
    )

    memory.write(
        addr_matrix_b,
        [
            5.0, 6.0,
            7.0, 8.0
        ]
    )

    matmul_program = GPUProgram()

    matmul_program.add(
        Instruction(
            opcode=Opcode.MATMUL,
            dst=mem(addr_matrix_c),
            src1=mem(addr_matrix_a),
            src2=mem(addr_matrix_b),
            extra={
                "m": 2,
                "n": 2,
                "k": 2
            }
        )
    )

    matmul_program.add(
        Instruction(
            opcode=Opcode.HALT
        )
    )

    core.run(matmul_program)

    print(
        "C =",
        memory.read(addr_matrix_c, 4)
    )

    print()
    print("================================")
    print("SOFTMAX Test")
    print("================================")

    addr_softmax_in = memory.allocate(3)
    addr_softmax_out = memory.allocate(3)

    memory.write(
        addr_softmax_in,
        [1.0, 2.0, 3.0]
    )

    softmax_program = GPUProgram()

    softmax_program.add(
        Instruction(
            opcode=Opcode.SOFTMAX,
            dst=mem(addr_softmax_out),
            src1=mem(addr_softmax_in),
            size=3
        )
    )

    softmax_program.add(
        Instruction(
            opcode=Opcode.HALT
        )
    )

    core.run(softmax_program)

    print(
        "Softmax =",
        memory.read(addr_softmax_out, 3)
    )
