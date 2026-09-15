# gpu_isa.py
#
# Simple Virtual GPU Instruction Set Architecture (ISA)
#
# Purpose:
#   - Define a small instruction set for the homemade GPU
#   - Represent GPU instructions in a simple Python data structure
#   - Provide opcodes used later by gpu_compute.py
#
# Dependencies:
#   gpu_memory.py
#
# Design policy:
#   gpu_isa.py defines "what instructions exist"
#   gpu_compute.py defines "how instructions are executed"

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List, Union


# ============================================================
# Opcode Definition
# ============================================================

class Opcode(Enum):
    """
    Virtual GPU instruction opcodes.
    """

    # Memory
    LOAD = auto()
    STORE = auto()

    # Register / data movement
    MOV = auto()

    # Arithmetic
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    FMA = auto()

    # Matrix / Tensor operations
    MATMUL = auto()

    # Activation / reduction
    SOFTMAX = auto()

    # Control
    NOP = auto()
    HALT = auto()


# ============================================================
# Operand Type
# ============================================================

class OperandType(Enum):
    """
    Types of operands used by instructions.
    """

    REGISTER = auto()
    MEMORY = auto()
    IMMEDIATE = auto()


# ============================================================
# Operand
# ============================================================

@dataclass
class Operand:
    """
    Operand used by a GPU instruction.

    Examples
    --------
    Register:
        Operand(OperandType.REGISTER, 0)

    Memory address:
        Operand(OperandType.MEMORY, 100)

    Immediate value:
        Operand(OperandType.IMMEDIATE, 3.14)
    """

    type: OperandType
    value: Union[int, float]


# ============================================================
# Instruction
# ============================================================

@dataclass
class Instruction:
    """
    One virtual GPU instruction.

    Parameters
    ----------
    opcode:
        Operation code.

    dst:
        Destination operand.

    src1:
        First source operand.

    src2:
        Second source operand.

    src3:
        Third source operand.
        Mainly used by FMA.

    size:
        Number of elements for vector / matrix operations.

    extra:
        Optional metadata.
    """

    opcode: Opcode

    dst: Optional[Operand] = None
    src1: Optional[Operand] = None
    src2: Optional[Operand] = None
    src3: Optional[Operand] = None

    size: int = 1

    extra: Optional[dict] = None


# ============================================================
# Helper Functions
# ============================================================

def reg(index: int) -> Operand:
    """
    Create a register operand.

    Example:
        reg(0) -> R0
    """
    return Operand(
        OperandType.REGISTER,
        index
    )


def mem(address: int) -> Operand:
    """
    Create a memory operand.

    Example:
        mem(100) -> memory address 100
    """
    return Operand(
        OperandType.MEMORY,
        address
    )


def imm(value: Union[int, float]) -> Operand:
    """
    Create an immediate operand.

    Example:
        imm(1.0)
    """
    return Operand(
        OperandType.IMMEDIATE,
        value
    )


# ============================================================
# ISA Validation
# ============================================================

def validate_instruction(inst: Instruction) -> None:
    """
    Basic validation of an instruction.

    Raises
    ------
    ValueError
        If the instruction format is invalid.
    """

    if inst.size <= 0:
        raise ValueError(
            f"Instruction size must be > 0: {inst.size}"
        )

    # --------------------------------------------------------
    # LOAD
    #
    # LOAD R0, [100]
    # --------------------------------------------------------

    if inst.opcode == Opcode.LOAD:

        if inst.dst is None or inst.src1 is None:
            raise ValueError(
                "LOAD requires dst and src1."
            )

        if inst.dst.type != OperandType.REGISTER:
            raise ValueError(
                "LOAD destination must be a register."
            )

        if inst.src1.type != OperandType.MEMORY:
            raise ValueError(
                "LOAD source must be memory."
            )

    # --------------------------------------------------------
    # STORE
    #
    # STORE [100], R0
    # --------------------------------------------------------

    elif inst.opcode == Opcode.STORE:

        if inst.dst is None or inst.src1 is None:
            raise ValueError(
                "STORE requires dst and src1."
            )

        if inst.dst.type != OperandType.MEMORY:
            raise ValueError(
                "STORE destination must be memory."
            )

        if inst.src1.type != OperandType.REGISTER:
            raise ValueError(
                "STORE source must be a register."
            )

    # --------------------------------------------------------
    # MOV
    # --------------------------------------------------------

    elif inst.opcode == Opcode.MOV:

        if inst.dst is None or inst.src1 is None:
            raise ValueError(
                "MOV requires dst and src1."
            )

        if inst.dst.type != OperandType.REGISTER:
            raise ValueError(
                "MOV destination must be a register."
            )

    # --------------------------------------------------------
    # Binary arithmetic
    #
    # ADD R0, R1, R2
    # --------------------------------------------------------

    elif inst.opcode in (
        Opcode.ADD,
        Opcode.SUB,
        Opcode.MUL,
        Opcode.DIV
    ):

        if (
            inst.dst is None or
            inst.src1 is None or
            inst.src2 is None
        ):
            raise ValueError(
                f"{inst.opcode.name} requires "
                f"dst, src1 and src2."
            )

        if inst.dst.type != OperandType.REGISTER:
            raise ValueError(
                f"{inst.opcode.name} destination "
                "must be a register."
            )

    # --------------------------------------------------------
    # FMA
    #
    # dst = src1 * src2 + src3
    # --------------------------------------------------------

    elif inst.opcode == Opcode.FMA:

        if (
            inst.dst is None or
            inst.src1 is None or
            inst.src2 is None or
            inst.src3 is None
        ):
            raise ValueError(
                "FMA requires dst, src1, src2, src3."
            )

        if inst.dst.type != OperandType.REGISTER:
            raise ValueError(
                "FMA destination must be a register."
            )

    # --------------------------------------------------------
    # MATMUL
    # --------------------------------------------------------

    elif inst.opcode == Opcode.MATMUL:

        if (
            inst.dst is None or
            inst.src1 is None or
            inst.src2 is None
        ):
            raise ValueError(
                "MATMUL requires dst, src1 and src2."
            )

        if inst.extra is None:
            raise ValueError(
                "MATMUL requires matrix shape metadata "
                "in extra."
            )

        required_keys = {
            "m",
            "n",
            "k"
        }

        if not required_keys.issubset(inst.extra.keys()):
            raise ValueError(
                "MATMUL extra must contain m, n and k."
            )

    # --------------------------------------------------------
    # SOFTMAX
    # --------------------------------------------------------

    elif inst.opcode == Opcode.SOFTMAX:

        if inst.dst is None or inst.src1 is None:
            raise ValueError(
                "SOFTMAX requires dst and src1."
            )

    # --------------------------------------------------------
    # NOP / HALT
    # --------------------------------------------------------

    elif inst.opcode in (
        Opcode.NOP,
        Opcode.HALT
    ):
        pass

    else:
        raise ValueError(
            f"Unknown opcode: {inst.opcode}"
        )


# ============================================================
# Program
# ============================================================

class GPUProgram:
    """
    Container for a list of virtual GPU instructions.
    """

    def __init__(self):
        self.instructions: List[Instruction] = []

    def add(self, instruction: Instruction) -> None:
        """
        Add an instruction after validation.
        """

        validate_instruction(instruction)

        self.instructions.append(
            instruction
        )

    def __len__(self) -> int:
        return len(self.instructions)

    def __getitem__(self, index: int) -> Instruction:
        return self.instructions[index]

    def dump(self) -> None:
        """
        Print the program in readable form.
        """

        for pc, inst in enumerate(self.instructions):

            print(
                f"{pc:04d}: "
                f"{format_instruction(inst)}"
            )


# ============================================================
# Formatting
# ============================================================

def format_operand(
    operand: Optional[Operand]
) -> str:

    if operand is None:
        return ""

    if operand.type == OperandType.REGISTER:
        return f"R{operand.value}"

    if operand.type == OperandType.MEMORY:
        return f"[{operand.value}]"

    if operand.type == OperandType.IMMEDIATE:
        return f"{operand.value}"

    return "?"


def format_instruction(
    inst: Instruction
) -> str:

    op = inst.opcode.name

    operands = []

    for operand in (
        inst.dst,
        inst.src1,
        inst.src2,
        inst.src3
    ):
        if operand is not None:
            operands.append(
                format_operand(operand)
            )

    text = op

    if operands:
        text += " " + ", ".join(operands)

    if inst.size != 1:
        text += f"  ; size={inst.size}"

    if inst.extra:
        text += f"  ; {inst.extra}"

    return text


# ============================================================
# Simple Test
# ============================================================

if __name__ == "__main__":

    program = GPUProgram()

    # --------------------------------------------------------
    # Example:
    #
    # R0 = memory[0]
    # R1 = memory[1]
    # R2 = R0 + R1
    # memory[2] = R2
    # --------------------------------------------------------

    program.add(
        Instruction(
            opcode=Opcode.LOAD,
            dst=reg(0),
            src1=mem(0)
        )
    )

    program.add(
        Instruction(
            opcode=Opcode.LOAD,
            dst=reg(1),
            src1=mem(1)
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
            dst=mem(2),
            src1=reg(2)
        )
    )

    program.add(
        Instruction(
            opcode=Opcode.HALT
        )
    )

    print("=== GPU Program ===")

    program.dump()

    print()

    # --------------------------------------------------------
    # FMA example
    #
    # R3 = R0 * R1 + R2
    # --------------------------------------------------------

    fma_inst = Instruction(
        opcode=Opcode.FMA,
        dst=reg(3),
        src1=reg(0),
        src2=reg(1),
        src3=reg(2)
    )

    print(
        "FMA:",
        format_instruction(fma_inst)
    )

    # --------------------------------------------------------
    # MATMUL example
    #
    # C[M,N] = A[M,K] @ B[K,N]
    # --------------------------------------------------------

    matmul_inst = Instruction(
        opcode=Opcode.MATMUL,
        dst=mem(200),
        src1=mem(100),
        src2=mem(150),
        extra={
            "m": 2,
            "n": 3,
            "k": 4
        }
    )

    validate_instruction(matmul_inst)

    print(
        "MATMUL:",
        format_instruction(matmul_inst)
    )
