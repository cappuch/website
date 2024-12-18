# 8 Bit CPU and Assembler
I've been working on a simple 8-bit CPU and assembler for the past few days. I wanted to create a simple CPU that I could use to learn more about computer architecture and assembly language. I also wanted to create a simple assembler that I could use to write programs for the CPU. I've made some progress, and I wanted to share it with you.

## The CPU

### Specifications
- **Registers**:
  - `A` and `B`: General-purpose 8-bit registers.
  - `PC` (Program Counter): Points to the next instruction in memory.
- **Memory**:
  - `memory`: 1KB of memory, all bits are set to zero.
- **Flags**:
  - `zero_flag`: Set when the result of an operation is zero.
  - `carry_flag`: Set when an operation causes a carry or borrow.

### Supported Instructions
- `NOP` | `0x00`: No operation.
- `LDA` | `0x01`: Load the value at the address in the next byte into register `A`.
- `ADD` | `0x02`: Add the value at the address in the next byte to register `A`.
- `SUB` | `0x03`: Subtract the value at the address in the next byte from register `A`.
- `STA` | `0x04`: Store the value in register `A` at the address in the next byte.
- `JMP` | `0x05`: Jump to the address in the next byte.
- `JZ` | `0x06`: Jump to the address in the next byte if the zero flag is set.
- `HALT` | `0xFF`: Halt the CPU.


## The Assembler
Like any assembler, it converts assembly code into machine code that the CPU can execute. Each instruction is mapped to a unique opcode (e.g., LDA maps to 0x01). The assembler reads an assembly file, tokenizes it, and generates a machine code file that the CPU can execute.
The assembler supports two instruction sets:
- CPU Instructions: Basic operations like `NOP`, `LDA`, `ADD`, and `HALT`.
- GPU Instructions: Graphics-related operations like `SETX`, `PLOT`, and `RECT`.

The assembler actually supports directives to select the mode between CPU and GPU instructions. The directives are `.CPU` and `.GPU`. The assembler will only assemble instructions that are in the selected mode.

### Example Assembly Code
```
.CPU
START: LDA 0x10
       ADD 0x20
       STA 0x30
       HALT
.GPU
       SETX 0x05
       SETY 0x10
       SETC 0x02
       PLOT
       GHALT
```

But wait. I never mentioned a GPU. What's going on here?

## The GPU
It's a simple simulated graphics processor. It allows native Pygame rendering, which is exactly why I chose it.

### Specifications
- Resolution: 640x480 pixels.
- Memory:
  - VRAM: A flat array representing the screen's pixel data. Each pixel is stored as three consecutive bytes (R, G, B), resulting in a total memory size of 640 × 480 × 3 bytes.

- Signals:
  - HSYNC and VSYNC: Simulate horizontal and vertical synchronization signals, essential for VGA-style displays.
  - RGB Signals: Represent the current pixel's color output as normalized values (0.0 to 1.0).

- Registers:
  - Current X and Y Coordinates: Track the position of the pixel being processed.
  - Current RGB Color: Hold the active drawing color.

### Supported Instructions
- NOP (0x00): No operation. The GPU simply advances to the next instruction.
- SETX (0x01): Set the current X coordinate for drawing.
- SETY (0x02): Set the current Y coordinate for drawing.
- SETC (0x03): Set the current drawing color (R, G, B).
- PLOT (0x04): Plot a pixel at the current (X, Y) position with the current color.
- CLEAR (0x05): Clear the screen, resetting all pixels to black.
- LINE (0x06): Draw a line between two points using Bresenham's algorithm.
- RECT (0x07): Draw a filled rectangle with the current color.
- HALT (0xFF): Halt execution of the GPU program.
  

And did I forget to mention the language I wrote for it?

## The Language (Luaython)
I'm terrible at names. Best name I could have come up with at the time. It's similar to Lua and Python, but it's not either of them.

It's very, very basic.

For example, here's a test program:
```
compute
    var x = 5
    var y = 10
    var z = 0
    z = x + y

draw
    clear
    setpos 10 10
    setcolor 1
    rect 20 30
```

This directly translates to the following assembly code:
```
.CPU
LDA 5
STA 20
LDA 10
STA 21
LDA 0
STA 22
.GPU
CLEAR
SETX 10
SETY 10
SETC 1
RECT 20 30
GHALT
```
Which is then assembled into the following machine code:
```
0x01
0x05
0x04
0x14
0x01
0x0a
0x04
0x15
0x01
0x00
0x04
0x16
0x05
0x01
0x0a
0x02 
0x0a
0x03
0x01
0x07
0x14
0x1e
0xff
```

The language has two blocks: `compute` and `draw`.
The `compute` block is for CPU instructions, and the `draw` block is for GPU instructions. The language is compiled to assembly code, which is then assembled into machine code. It's simple, but it works.

I'm still working on it, but I'm happy with the progress I've made so far. I'll keep you updated on my progress.

If you want to try it out, you can access the web demos [here](https://cappuch.github.io/emulator/). (Assembler and Language only)