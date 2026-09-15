# gpu_memory.py
#
# Simple Virtual GPU Memory Model
#
# Purpose:
#   - Simulate GPU global memory in Python
#   - Allocate / free memory blocks
#   - Read / write scalar and array data
#   - Provide a base layer for gpu_isa.py and tensor.py
#
# This is intentionally simple for educational use.

from dataclasses import dataclass
from typing import List, Dict, Iterable


@dataclass
class MemoryBlock:
    """
    Information about one allocated memory block.
    """
    address: int
    size: int


class GPUMemory:
    """
    Very small virtual GPU global-memory model.

    Memory is represented as a Python list of float values.

    Example
    -------
    mem = GPUMemory(1024)

    addr = mem.allocate(4)
    mem.write(addr, [1.0, 2.0, 3.0, 4.0])

    print(mem.read(addr, 4))
    """

    def __init__(self, size: int = 1024):
        if size <= 0:
            raise ValueError("Memory size must be greater than 0.")

        self.size = size

        # Simulated GPU memory
        self.memory: List[float] = [0.0] * size

        # Next unused memory location.
        # Version 1 uses a simple linear allocator.
        self.next_address = 0

        # address -> MemoryBlock
        self.allocations: Dict[int, MemoryBlock] = {}

    # ------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------

    def allocate(self, size: int) -> int:
        """
        Allocate 'size' elements and return the starting address.
        """

        if size <= 0:
            raise ValueError("Allocation size must be greater than 0.")

        if self.next_address + size > self.size:
            raise MemoryError(
                f"GPU memory exhausted: "
                f"requested={size}, "
                f"available={self.size - self.next_address}"
            )

        address = self.next_address

        self.allocations[address] = MemoryBlock(
            address=address,
            size=size
        )

        self.next_address += size

        return address

    def free(self, address: int) -> None:
        """
        Mark an allocated block as freed.

        Note:
            This first version does not reuse freed regions yet.
            Memory reuse can be added later.
        """

        if address not in self.allocations:
            raise ValueError(
                f"Invalid or already freed address: {address}"
            )

        block = self.allocations[address]

        # Clear memory for easier debugging.
        for i in range(block.size):
            self.memory[address + i] = 0.0

        del self.allocations[address]

    # ------------------------------------------------------------
    # Read / Write
    # ------------------------------------------------------------

    def write(self, address: int, values: Iterable[float]) -> None:
        """
        Write multiple values beginning at 'address'.
        """

        values = list(values)

        self._check_range(address, len(values))

        for i, value in enumerate(values):
            self.memory[address + i] = float(value)

    def read(self, address: int, size: int) -> List[float]:
        """
        Read 'size' elements beginning at 'address'.
        """

        self._check_range(address, size)

        return self.memory[address:address + size]

    def write_scalar(self, address: int, value: float) -> None:
        """
        Write one scalar value.
        """

        self._check_range(address, 1)
        self.memory[address] = float(value)

    def read_scalar(self, address: int) -> float:
        """
        Read one scalar value.
        """

        self._check_range(address, 1)
        return self.memory[address]

    # ------------------------------------------------------------
    # Memory Operations
    # ------------------------------------------------------------

    def copy(
        self,
        dst_address: int,
        src_address: int,
        size: int
    ) -> None:
        """
        Copy data inside GPU memory.
        """

        values = self.read(src_address, size)
        self.write(dst_address, values)

    def fill(
        self,
        address: int,
        size: int,
        value: float
    ) -> None:
        """
        Fill a memory region with one value.
        """

        self._check_range(address, size)

        for i in range(size):
            self.memory[address + i] = float(value)

    # ------------------------------------------------------------
    # Debug / Information
    # ------------------------------------------------------------

    def dump(
        self,
        start: int = 0,
        size: int = 16
    ) -> None:
        """
        Print part of GPU memory.
        """

        self._check_range(start, size)

        print("GPU Memory Dump")
        print("----------------")

        for i in range(start, start + size):
            print(f"[{i:04d}] = {self.memory[i]}")

    def info(self) -> None:
        """
        Show current memory usage.
        """

        allocated = sum(
            block.size
            for block in self.allocations.values()
        )

        print("GPU Memory Information")
        print("----------------------")
        print(f"Total      : {self.size}")
        print(f"Allocated  : {allocated}")
        print(f"Free       : {self.size - allocated}")
        print(f"Blocks     : {len(self.allocations)}")

    def reset(self) -> None:
        """
        Reset the whole GPU memory.
        """

        self.memory = [0.0] * self.size
        self.next_address = 0
        self.allocations.clear()

    # ------------------------------------------------------------
    # Internal Utilities
    # ------------------------------------------------------------

    def _check_range(
        self,
        address: int,
        size: int
    ) -> None:

        if address < 0:
            raise IndexError(
                f"Negative GPU memory address: {address}"
            )

        if size < 0:
            raise ValueError(
                f"Invalid size: {size}"
            )

        if address + size > self.size:
            raise IndexError(
                f"GPU memory access out of range: "
                f"address={address}, size={size}, "
                f"memory_size={self.size}"
            )


# ----------------------------------------------------------------
# Simple Test
# ----------------------------------------------------------------

if __name__ == "__main__":

    gpu_memory = GPUMemory(size=64)

    print("=== Allocate ===")

    addr_a = gpu_memory.allocate(4)
    addr_b = gpu_memory.allocate(4)

    print("A address:", addr_a)
    print("B address:", addr_b)

    print()

    print("=== Write ===")

    gpu_memory.write(
        addr_a,
        [1.0, 2.0, 3.0, 4.0]
    )

    gpu_memory.write(
        addr_b,
        [10.0, 20.0, 30.0, 40.0]
    )

    print("A =", gpu_memory.read(addr_a, 4))
    print("B =", gpu_memory.read(addr_b, 4))

    print()

    print("=== Copy ===")

    gpu_memory.copy(
        dst_address=addr_b,
        src_address=addr_a,
        size=4
    )

    print("B =", gpu_memory.read(addr_b, 4))

    print()

    print("=== Memory Info ===")

    gpu_memory.info()

    print()

    print("=== Dump ===")

    gpu_memory.dump(
        start=0,
        size=12
    )
