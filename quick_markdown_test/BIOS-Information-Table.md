<div id="header">

# NVIDIA BIOS Information Table Specification

</div>

<div id="preamble">

<div class="sectionbody">

</div>

</div>

## Purpose

<div class="sectionbody">

<div class="paragraph">

This document describes the BIOS Information Table (BIT), the top-level
description table in the NVIDIA VBIOS.

</div>

<div class="paragraph">

The BIT points to various code sections and data tables used by both the
BIOS and driver software. The tables typically contain GPU and
board-specific information.

</div>

<div class="paragraph">

VBIOS pointers may point to data beyond the end of the PC-compatible
(legacy BIOS, Code Type 00h) image. If a UEFI (Code Type 03h) image
follows the PC-compatible image, then the pointer must be adjusted to be
an offset into the data following the UEFi Image.

</div>

<div class="paragraph">

If (pointer \> PC-compatible image length) { adjusted\_pointer = pointer
+ UEFI image length }

</div>

<div class="paragraph">

A GPU firmware file (.ROM) may contain data for HW consumption preceding
the PCI Expansion ROM contents. The start of the PCI Expansion ROM can
be found by checking 512 byte boundaries for the {055h,0AAh} PCI
Expansion ROM signature. Additionally, the pointer to the PCI Data
Structure should be followed and checked for the "PCIR" PCI Data
Structure signature to confirm a valid PCI Expansion ROM has been found.
When reading the firmware in a system using MMIO the PCI Expansion ROM
will begin at PCI Expansion ROM BAR + offset 0.

</div>

### BIOS Information Table Structure

<div style="clear:left">

</div>

<div class="paragraph">

The BIT is a series of tokenized structures, beginning with a BIT
definition structure, and a series of BIT tokens and data pointers. The
data pointers point to a grouping of data items that are used by NVIDIA
software to locate, use, and/or modify device-specific data.

</div>

#### BIT Header

<div class="tableblock">

| Name          | Bit width |  Values  | Meaning                                                                             |
| :------------ | --------: | :------: | :---------------------------------------------------------------------------------- |
| ID            |        16 |  0xB8FF  | BIT Header Identifier                                                               |
| Signature     |        32 | "BIT\\0" | BIT Header Signature                                                                |
| BCD Version   |        16 |  0x0100  | BCD Version 1.00 (major version in the upper byte, minor version in the lower byte) |
| Header Size   |         8 |    12    | Size of BIT Header (in bytes)                                                       |
| Token Size    |         8 |    6     | Size of BIT Tokens (in bytes)                                                       |
| Token Entries |         8 |    ?     | Number of token entries that follow                                                 |
| Checksum      |         8 |    0     | BIT Header Checksum                                                                 |

</div>

</div>

## BIT Token Structure

<div class="sectionbody">

<div class="paragraph">

Each BIT token has the same format and length. Prior knowledge of the
data format, based on the data version indicated, is necessary to access
the actual data.

</div>

<div class="tableblock">

| Name         | Bit width | Values and Meaning                                                                                                                                   |
| :----------- | --------: | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
| ID           |         8 | Unique identifier indicating what data is pointed to by the Data Pointer                                                                             |
| Data Version |         8 | Version of the data structure pointed to by the Data Pointer                                                                                         |
| Data Size    |        16 | Size of data structure pointed to by the Data Pointer (in bytes)                                                                                     |
| Data Pointer |        16 | Pointer (offset) to the actual data structure. A NULL (0) pointer indicates no data exists for this token, and that it can be treated as a BIT\_NOP. |

</div>

### BIT Tokens

<div style="clear:left">

</div>

<div class="paragraph">

**Deprecated tokens and data structure versions are highlighted in
<span style="color: red;">red</span>.**

</div>

<div class="tableblock">

Name

</div>

</div>

ID

Meaning

Corresponding Data Structure

BIT\_TOKEN\_I2C\_PTRS

0x32 (‘2’)

I2C Script Pointers

[BIT\_I2C\_PTRS](#_bit_i2c_ptrs)

BIT\_TOKEN\_DAC\_PTRS

0x41 (‘A’)

DAC Data Pointers

[BIT\_DAC\_PTRS](#_bit_dac_ptrs)

BIT\_TOKEN\_BIOSDATA

0x42 (‘B’)

BIOS Data

BIOSDATA [<span style="color: red;">(Version
1)</span>](#BIT_BIOSDATA_v1) [(Version 2)](#BIT_BIOSDATA_v2)

BIT\_TOKEN\_CLOCK\_PTRS

0x43 (‘C’)

Clock Script Pointers

CLK PTRS [<span style="color: red;">(version
1)</span>](#BIT_CLOCK_PTRS_V1) [(version 2)](#BIT_CLOCK_PTRS_V2)

BIT\_TOKEN\_DFP\_PTRS

0x44 (‘D’)

DFP/Panel Data Pointers

[BIT\_DFP\_PTRS](#_bit_dfp_ptrs)

BIT\_TOKEN\_NVINIT\_PTRS

0x49 (‘I’)

Initialization Table Pointers

[BIT\_NVINIT\_PTRS](#_bit_nvinit_ptrs)

BIT\_TOKEN\_LVDS\_PTRS

0x4C (‘L’)

LVDS Table Pointers

[BIT\_LVDS\_PTRS](#_bit_lvds_ptrs)

BIT\_TOKEN\_MEMORY\_PTRS

0x4D (‘M’)

Memory Control/Programming Pointers

BIT\_MEMORY\_PTRS [(<span style="color: red;">Version
1</span>)](#BIT_MEMORY_PTRS_v1) [(Version 2)](#BIT_MEMORY_PTRS_v2)

BIT\_TOKEN\_NOP

0x4E (‘N’)

No Operation

[BIT\_NOP](#_bit_nop)

BIT\_TOKEN\_PERF\_PTRS

0x50 (‘P’)

Performance Table Pointers

BIT\_PERF\_PTRS [(<span style="color: red;">Version
1</span>)](#BIT_PERF_PTRS_v1) [(Version 2)](#BIT_PERF_PTRS_v2)

BIT\_TOKEN\_STRING\_PTRS

0x53 (‘S’)

String Pointers

BIT\_STRING\_PTRS [(<span style="color: red;">Version
1</span>)](#BIT_STRING_PTRS_v1) [(Version 2)](#BIT_STRING_PTRS_v2)

BIT\_TOKEN\_TMDS\_PTRS

0x54 (‘T’)

TMDS Table Pointers

[BIT\_TMDS\_PTRS](#_bit_tmds_ptrs)

BIT\_TOKEN\_DISPLAY\_PTRS

0x55 (‘U’)

Display Control/Programming Pointers

[BIT\_DISPLAY\_PTRS](#_bit_display_ptrs)

BIT\_TOKEN\_VIRTUAL\_PTRS

0x56 (‘V’)

Virtual Field Pointers

[BIT\_VIRTUAL\_PTRS](#_bit_virtual_ptrs)

BIT\_TOKEN\_32BIT\_PTRS

0x63 (‘c’)

32-bit Pointer Data

[BIT\_32BIT\_PTRS](#_bit_32bit_ptrs)

BIT\_TOKEN\_DP\_PTRS

0x64 (‘d’)

DP Table Pointers

[BIT\_DP\_PTRS](#_bit_dp_ptrs)

BIT\_TOKEN\_FALCON\_DATA

0x70 (‘p’)

Falcon Ucode Data

PMU Table Pointers: [BIT\_FALCON\_DATA](#BIT_FALCON_DATA_v2) or
[<span style="color: red;">BIT\_PMU\_PTRS</span>](#BIT_PMU_PTRS_v1)

BIT\_TOKEN\_UEFI\_DATA

0x75 (‘u’)

UEFI Driver Data

[BIT\_UEFI\_DATA](#_bit_uefi_data)

BIT\_TOKEN\_MXM\_DATA

0x78 (‘x’)

MXM Configuration Data

[BIT\_MXM\_DATA](#_bit_mxm_data)

BIT\_TOKEN\_BRIDGE\_FW\_DATA

0x52 (‘R’)

Bridge Firmware Data

[BIT\_BRIDGE\_FW\_DATA](#_bit_bridge_fw_data)

## Parsing Rules

<div class="sectionbody">

<div class="paragraph">

The [BIT header](#_bit%20header) should be searched for as follows:

</div>

<div class="ulist">

  - ID plus Signature should be used to locate the BIT structure. Once
    found, the data immediately following the BIT header is the first
    token.

  - HeaderChecksum is a 0 checksum of the entire BIT header. The correct
    checksum can be found by adding BIT\_Header.HeaderSize consecutive
    bytes together, starting with the first byte of the BIT header. A
    valid BIT will provide a byte sum of 00h.

  - HeaderSize contains a value that indicates how big the actual BIT
    header is. The first token can be found by adding this value to the
    start address of the BIT header.

  - TokenSize indicates how big each token entry is. All tokens are the
    same size.

  - TokenEntries indicates how many tokens are contained in the list and
    should be processed by software.

</div>

</div>

## BIT Data Structures

<div class="sectionbody">

<div class="paragraph">

**Deprecated data structure versions are highlighted in
<span style="color: red;">red</span>.**

</div>

### BIT\_I2C\_PTRS

<div style="clear:left">

</div>

<div class="paragraph">

This data structure contains I2C scripting data.

</div>

<div class="tableblock">

| Name         | Bit width | Values and Meaning                                                       |
| :----------- | --------: | :----------------------------------------------------------------------- |
| I2CScripts   |        16 | Pointer to the I2C Scripts table                                         |
| ExtHWMonInit |        16 | Pointer to an I2C script used to initialize an external hardware monitor |

</div>

### BIT\_DAC\_PTRS

<div style="clear:left">

</div>

<div class="paragraph">

This data structure contains DAC related data.

</div>

<div class="tableblock">

| Name       | Bit width | Values and Meaning          |
| :--------- | --------: | :-------------------------- |
| DACDataPtr |        16 | Pointer to DAC related data |
| DACFlags   |         8 | DAC Flags                   |

</div>

<div class="ulist">

<div class="title">

DACFlags

</div>

  - Bit 0:0 - DAC Sleep Mode Support (via NV\_PDISP\_DAC\_TEST).
    Possible values are:
    
    <div class="ulist">
    
      - 0x0 - Not Supported
    
      - 0x1 - Supported
    
    </div>

  - Bits 7:1 - Reserved

</div>

### BIT\_BIOSDATA (<span style="color: red;">Version 1</span>)

<div style="clear:left">

</div>

<div class="paragraph">

**<span style="color: red;">This data structure has been
deprecated.</span>** It contains BIOS related data.

</div>

<div class="tableblock">

| Name                   | Bit width | Values and Meaning                             |
| :--------------------- | --------: | :--------------------------------------------- |
| BIOS Version           |        32 | BIOS Binary Version                            |
| BIOS OEM Version       |         8 | BIOS OEM Version Number                        |
| BIOS Checksum          |         8 | BIOS 0 Checksum inserted during the build      |
| INT15 POST Callbacks   |        16 | INT15 Callbacks issued during POST             |
| INT15 SYSTEM Callbacks |        16 | General INT15 Callbacks                        |
| BIOS Board ID          |        16 | Board ID                                       |
| Frame Count            |        16 | Number of frames to display the SignOn Message |
| BIOSMOD Date           |        24 | Date BIOSMod was last run (in MMDDYY format)   |

</div>

<div class="ulist">

<div class="title">

INT15 POST Callbacks

</div>

  - Bit 0:0 - Get Panel ID

  - Bit 1:1 - Get TV Format (NTSC/PAL/NTSC-J/etc.)

  - Bit 2:2 - Get Boot Device

  - Bit 3:3 - Get Panel Expansion/Centering

  - Bit 4:4 - Perform POST Complete Callback

  - Bit 5:5 - Get RAM Configuration (OEM Specific, deprecated)

  - Bit 6:6 - Get TV Connection Type (SVIDEO/Composite/etc.)

  - Bit 7:7 - OEM External Initialization

  - Bits 15:8 - Reserved

</div>

<div class="ulist">

<div class="title">

INT15 SYSTEM Callbacks

</div>

  - Bit 0:0 - Make DPMS Bypass Callback

  - Bit 1:1 - Get TV Format Callback (NTSC/PAL/etc.)

  - Bit 2:2 - Make Spread Spectrum Bypass Callback

  - Bit 3:3 - Make Display Switch Bypass Callback

  - Bit 4:4 - Make Device Control Setting Bypass Callback

  - Bit 5:5 - Make DDC Call Bypass Callback

  - Bit 6:6 - Make DFP Center/Expand Bypass Callback

  - Bits 15:7 - Reserved

</div>

### BIT\_BIOSDATA (Version 2)

<div style="clear:left">

</div>

<div class="paragraph">

This data structure contains BIOS related data.

</div>

<div class="tableblock">

<table style="width:99%;">
<colgroup>
<col style="width: 24%" />
<col style="width: 3%" />
<col style="width: 72%" />
</colgroup>
<thead>
<tr class="header">
<th style="text-align: left;">Name</th>
<th style="text-align: right;">Bit width</th>
<th style="text-align: left;">Values and Meaning</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;"><p>BIOS Version</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>BIOS Binary Version</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>BIOS OEM Version</p></td>
<td style="text-align: right;"><p>8</p></td>
<td style="text-align: left;"><p>BIOS OEM Version Number</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>BIOS Checksum</p></td>
<td style="text-align: right;"><p>8</p></td>
<td style="text-align: left;"><p>BIOS 0 Checksum inserted during the build</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>INT15 POST Callbacks</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>INT15 Callbacks during POST</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>INT15 SYSTEM Callbacks</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>General INT15 Callbacks</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Frame Count</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>Number of frames to display SignOn Message</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Reserved</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Reserved</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Max Heads at POST</p></td>
<td style="text-align: right;"><p>8</p></td>
<td style="text-align: left;"><p>Max number of heads to boot at POST</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Memory Size Report (MSR)</p></td>
<td style="text-align: right;"><p>8</p></td>
<td style="text-align: left;"><p>Scheme for computing memory size displayed in Control Panel. Does not affect functionality in any way</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>hScale Factor</p></td>
<td style="text-align: right;"><p>8</p></td>
<td style="text-align: left;"><p>Horizontal Scale Factor</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>vScale Factor</p></td>
<td style="text-align: right;"><p>8</p></td>
<td style="text-align: left;"><p>Vertical Scale Factor</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p><a href="#_data_range_table">Data Range Table</a> Pointer</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>Pointer to the table of pointers identifying where all data in the VGA BIOS image is located that the OS or EFI GPU driver need</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>ROMpacks Pointer</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>Pointer to any ROMpacks. A NULL (0) pointer indicates that no run-time ROMpacks are present<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Applied ROMpacks Pointer</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>Pointer to a list of indexes of applied run-time ROMpacks</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Applied ROMpack Max</p></td>
<td style="text-align: right;"><p>8</p></td>
<td style="text-align: left;"><p>Maximum number of stored indexes in the list pointed to by the Applied ROMpacks pointer</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Applied ROMpack Count</p></td>
<td style="text-align: right;"><p>8</p></td>
<td style="text-align: left;"><p>Number of applied run-time ROMpacks<br />
NOTE: Count can be higher than amount stored at the AppliedROMpacksPtr array, if more than the value at AppliedROMpackMax were applied</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Module Map External 0</p></td>
<td style="text-align: right;"><p>8</p></td>
<td style="text-align: left;"><p>Module Map External 0 byte. Indicates whether modules outside of the BIT and not at fixed addresses are included in the binary</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Compression Info Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to compression information structure (for use only by stage0 build script and decompression run-time code)</p></td>
</tr>
</tbody>
</table>

</div>

<div class="ulist">

<div class="title">

INT15 POST Callbacks

</div>

  - Bit 0:0 - Get Panel ID

  - Bit 1:1 - Get TV Format (NTSC/PAL/NTSC-J/etc.)

  - Bit 2:2 - Get Boot Device

  - Bit 3:3 - Get Panel Expansion/Centering

  - Bit 4:4 - Perform POST Complete Callback

  - Bit 5:5 - Get RAM Configuration (OEM Specific – should be obsolete)

  - Bit 6:6 - Get TV Connection Type (SVIDEO/Composite/etc.)

  - Bit 7:7 - OEM External Initialization

  - Bits 15:8 - Reserved

</div>

<div class="ulist">

<div class="title">

INT15 SYSTEM Callbacks

</div>

  - Bit 0:0 - Make DPMS Bypass Callback

  - Bit 1:1 - Get TV Format Callback (NTSC/PAL/etc.)

  - Bit 2:2 - Make Spread Spectrum Bypass Callback

  - Bit 3:3 - Make Display Switch Bypass Callback

  - Bit 4:4 - Make Device Control Setting Bypass Callback

  - Bit 5:5 - Make DDC Call Bypass Callback

  - Bit 6:6 - Make DFP Center/Expand Bypass Callback

  - Bits 15:7 - Reserved

</div>

<div class="ulist">

<div class="title">

Module Map External 0

</div>

  - Bit 0:0 - Underflow and Error Reporting. This mode enables HW to
    red-fill the screen on display pipe underflow, and causes the VBIOS
    to make the overscan border red on poll timeouts, as well as FB
    pattern test failures. **<span style="color: red;">This mode should
    never be enabled on production VBIOSes\!</span>**

  - Bit 1:1 - Coproc Build. Set when a VBIOS is intended to work as a
    coprocessor and does not support any displays  

  - Bit 2:2 - Reserved

  - Bit 3:3 - Reserved

  - Bit 4:4 - Reserved

</div>

### Data Range Table

<div style="clear:left">

</div>

<div class="paragraph">

The Data Table contains pointers identifying where all data in the VGA
BIOS image are located that the OS GPU drivers or EFI GPU driver needs.

</div>

<div class="ulist">

  - Only data in the x86 code type PCI firmware block are included in
    the Data Range Table.

  - Any other PCI firmware blocks present are defined to be "all data"
    (or in the case of an EFI PCI firmware block, all code).

</div>

<div class="tableblock">

| Name                | Bit width | Values and Meaning                                   |
| :------------------ | --------: | :--------------------------------------------------- |
| Image Start         |        16 | Pointer to the start of the binary image (0x0000)    |
| BIT End             |        16 | Pointer to the end of the BIOS Information Table     |
| Data Resident Start |        16 | Pointer to the start of the resident data section    |
| Data Resident End   |        16 | Pointer to the end of the resident data section      |
| Data Discard Start  |        16 | Pointer to the start of the discardable data section |
| Data Discard End    |        16 | Pointer to the end of the discardable data section   |
| End of List         |        32 | End of the list (0x0000, 0x0000)                     |

</div>

### BIT\_CLOCK\_PTRS (<span style="color: red;">Version 1</span>)===

<div style="clear:left">

</div>

<div class="paragraph">

<span style="color: red;">THIS STRUCTURE VERSION IS NOW DEPRECATED.
PLEASE REFER TO VERSION 2 BELOW.</span>

</div>

<div class="paragraph">

This data structure contains data related to clock programming.

</div>

<div class="tableblock">

| Name                       | Bit width | Values and Meaning                               |
| :------------------------- | --------: | :----------------------------------------------- |
| PLL Register Table Pointer |        32 | Pointer to the table of PLL registers            |
| Clock Script               |        32 | Pointer to a script to run after changing clocks |
| PLL Info Table Pointer     |        16 | Pointer to the PLL info table                    |
| Clock Frequency Table      |        32 | Pointer to the fixed clock frequency table       |
| FIFO Table                 |        16 | Pointer to the DAC/CRTC FIFO settings table      |
| Noise-Aware PLL Table      |        16 | Pointer to the noise-aware PLL yable             |

</div>

### BIT\_CLOCK\_PTRS (Version 2)===

<div style="clear:left">

</div>

<div class="paragraph">

This data structure contains Clock Programming related data.

</div>

<div class="tableblock">

| PLL Info Table Pointer             | 32 | Pointer to PLL Info Table             |
| :--------------------------------- | -: | :------------------------------------ |
| VBE Mode PCLK table                | 32 | Pointer to VBE Mode PCLK Table        |
| Clocks Table Pointer               | 32 | Pointer to Clocks Table               |
| Clock Programming Table Pointer    | 32 | Pointer to Clock Programming Table    |
| NAFLL Table Pointer                | 32 | Pointer to NAFLL Table                |
| ADC Table Pointer                  | 32 | Pointer to ADC Table                  |
| Frequency Controller Table Pointer | 32 | Pointer to Frequency Controller Table |

</div>

### BIT\_DFP\_PTRS

<div style="clear:left">

</div>

<div class="paragraph">

This data structure contains data related to DFP programming.

</div>

<div class="tableblock">

| Name             | Bit width | Values and Meaning                                   |
| :--------------- | --------: | :--------------------------------------------------- |
| FP Established   |        16 | Pointer to a table of VESA Established Timing tables |
| FP Table Pointer |        16 | Pointer to the VBIOS-internal flat panel tables      |

</div>

### BIT\_NVINIT\_PTRS

<div style="clear:left">

</div>

<div class="paragraph">

This data structure contains initialization table pointers.

</div>

<div class="tableblock">

<table style="width:99%;">
<colgroup>
<col style="width: 24%" />
<col style="width: 3%" />
<col style="width: 72%" />
</colgroup>
<thead>
<tr class="header">
<th style="text-align: left;">Name</th>
<th style="text-align: right;">Bit width</th>
<th style="text-align: left;">Values and Meaning</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;"><p>Init Script Table Pointer</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>Pointer to the table of Devinit script pointers</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Macro Index Table Pointer</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>Pointer to the macro index table</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Macro Table Pointer</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>Pointer to the macro table</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Condition Table Pointer</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>Pointer to a table of Devinit conditionals used with the INIT_CONDITION opcode</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>I/O Condition Table Pointer</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>Pointer to a table of Devinit I/O conditionals used with the INIT_IO_CONDITION opcode</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>I/O Flag Condition Table Pointer</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>Pointer to a table of Devinit I/O conditionals used with the INIT_IO_FLAG_CONDITION opcode</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Init Function Table Pointer</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>Pointer to the init function table</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>VBIOS Private Boot Script Pointer</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>Pointer to the VBIOS private boot script</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Data Arrays Table Pointer</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>Pointer to the data arrays table</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>PCIe Settings Script Pointer</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>Pointer to the PCIe settings script<br />
</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Devinit Tables Pointer</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>Pointer to the contiguous segment containing tables required by Devinit opcodes</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Devinit Tables Size</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>Size of the contiguous segment containing tables required by Devinit opcodes</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Boot Scripts Pointer</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>Pointer to the contiguous segment containing Devinit boot scripts</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Boot Scripts Size</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>Size of the contiguous segment containing Devinit boot scripts</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>NVLink Configuration Data Pointer</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>Pointer to NVLink Configuration Data<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Boot Scripts Non-GC6 Pointer</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>Pointer to the continuous section of devinit that is not required on GC6 exit</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Boot Scripts Size Non-GC6</p></td>
<td style="text-align: right;"><p>16</p></td>
<td style="text-align: left;"><p>Size of contiguous section containing devinit that is not required on GC6 exit</p></td>
</tr>
</tbody>
</table>

</div>

### BIT\_LVDS\_PTRS

<div style="clear:left">

</div>

<div class="paragraph">

This data structure contains LVDS Initialization table pointers.

</div>

<div class="tableblock">

| Name                    | Bit width | Values and Meaning             |
| :---------------------- | --------: | :----------------------------- |
| LVDS Info Table Pointer |        16 | Pointer to the LVDS info table |

</div>

### BIT\_MEMORY\_PTRS (<span style="color: red;">Version 1</span>)

<div style="clear:left">

</div>

<div class="paragraph">

**<span style="color: red;">Version 1 of this data structure has been
deprecated.</span>**

</div>

<div class="paragraph">

This data structure contains memory control/programming related pointers

</div>

<div class="tableblock">

| Name                                   | Bit width | Values and Meaning                            |
| :------------------------------------- | --------: | :-------------------------------------------- |
| Memory Reset Table Pointer             |        16 | Pointer to the memory reset script            |
| Memory Strap Data Count                |         8 | Memory strap data Count                       |
| Memory Strap Translation Table Pointer |        16 | Pointer to the memory strap translation table |
| Memory Data VREF On Pointer            |        16 | Pointer to the data VREF on script            |
| Memory Data DQS On Pointer             |        16 | Pointer to the data DQS on script             |
| Memory Data DLCELL On Pointer          |        16 | Pointer to the data DLCELL on script          |
| Memory Data DLCELL Off Pointer         |        16 | Pointer to the data DLCELL off script         |

</div>

### BIT\_MEMORY\_PTRS (Version 2)

<div style="clear:left">

</div>

<div class="paragraph">

This data structure contains pointers related to memory control and
programming.

</div>

<div class="tableblock">

| Name                                   | Bit width | Values and Meaning                                                                                           |
| :------------------------------------- | --------: | :----------------------------------------------------------------------------------------------------------- |
| Memory Strap Data Count                |         8 | Memory strap data count                                                                                      |
| Memory Strap Translation Table Pointer |        16 | Pointer to the memory strap translation table                                                                |
| Memory Information Table Pointer       |        16 | Pointer to the memory information table                                                                      |
| Reserved                               |        64 |                                                                                                              |
| Memory Partition Information Table     |        32 | Pointer to the memory partition information table                                                            |
| Memory Script List Pointer             |        32 | Pointer to Memory Script List, a list of 32-bit pointers to devinit scripts used to program FB register set. |

</div>

### BIT\_NOP

<div style="clear:left">

</div>

<div class="paragraph">

This data structure is a "no operation" indicator and contains no data.
BIT\_TOKEN\_NOP should be skipped by processing software, and processing
should continue at the next token.

</div>

### BIT\_PERF\_PTRS (<span style="color: red;">Version 1</span>)

<div style="clear:left">

</div>

<div class="paragraph">

**<span style="color: red;">Version 1 of this data structure has been
deprecated.</span>**

</div>

<div class="paragraph">

This data structure contains performance table pointers, which are
stored as 32-bit offsets to the data.

</div>

<div class="ulist">

  - These pointers are only used by system software, and may point at
    data outside the base 64K ROM image.

  - A conversion from Real Mode segment:offset format is done using the
    following algorithm: `(((16bit)SEGMENT) << 4) + ((16bit)OFFSET)`

</div>

<div class="tableblock">

<table style="width:99%;">
<colgroup>
<col style="width: 24%" />
<col style="width: 3%" />
<col style="width: 72%" />
</colgroup>
<thead>
<tr class="header">
<th style="text-align: left;">Name</th>
<th style="text-align: right;">Bit width</th>
<th style="text-align: left;">Values and Meaning</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;"><p>Performance Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the performance table</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Memory Tweak Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the memory tweak table</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Drive/Slew Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the drive/slew table</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Board Temperature Control Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to board temperature control limits</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>GPIO Voltage Select Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the GPIO voltage select table</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>AGP Clock Frequency</p></td>
<td style="text-align: right;"><p>8</p></td>
<td style="text-align: left;"><p>AGP clock frequency used for PCIe bus speed (in MHz)<br />
TODO: is this AGP or PCIe?</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>NVCLK Performance Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the NVCLK performance table</p></td>
</tr>
</tbody>
</table>

</div>

### BIT\_PERF\_PTRS (Version 2)

<div style="clear:left">

</div>

<div class="tableblock">

<table style="width:99%;">
<colgroup>
<col style="width: 24%" />
<col style="width: 3%" />
<col style="width: 72%" />
</colgroup>
<thead>
<tr class="header">
<th style="text-align: left;">Name</th>
<th style="text-align: right;">Bit width</th>
<th style="text-align: left;">Values and Meaning</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;"><p>Performance Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the performance table<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Memory Clock Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the memory clock table<br />
</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Memory Tweak Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the memory tweak table<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Power Control Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the power control table<br />
</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Thermal Control Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the thermal control table<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Thermal Device Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the thermal device table<br />
</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Thermal Coolers Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the thermal coolers table<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Performance Settings Script Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to a Devinit script containing performance-related settings<br />
See Note 1.</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Continuous Virtual Binning Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the continuous virtual binning table<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Ventura Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the Ventura table<br />
</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Power Sensors Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the power sensors table<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Power Policy Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the power policy table<br />
</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>P-State Clock Range Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the P-State clock range table<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Voltage Frequency Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the voltage frequency table<br />
</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Virtual P-State Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the virtual P-State table<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Power Topology Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the power topology table<br />
</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Power Leakage Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the power leakage table<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Performance Test Specifications Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the performance test specifications table<br />
</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Thermal Channel Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the thermal channel table<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Thermal Adjustment Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the thermal adjustment table<br />
</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Thermal Policy Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the thermal policy table<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>P-State Memory Clock Frequency Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the P-State memory clock frequency table<br />
</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Fan Cooler Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the fan cooler table<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Fan Policy Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to the fan policy table<br />
</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>DI/DT Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to DI/DT Table<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Fan Test Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to Fan Test Table<br />
</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Voltage Rail Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to Voltage Rail Table<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Voltage Device Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to Voltage Device Table<br />
</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Voltage Policy Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to Voltage Policy Table<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>LowPower Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to LowPower Table<br />
</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>LowPower PCIe Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to LowPower PCIe Table<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>LowPower PCIe-Platform Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to LowPower PCIe-Platform Table<br />
</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>LowPower GR Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to LowPower GR Table<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>LowPower MS Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to LowPower MS Table<br />
</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>LowPower DI Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to LowPower DI Table<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>LowPower GC6 Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to LowPower GC6 Table<br />
</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>LowPower PSI Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to LowPower PSI Table<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Thermal Monitor Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to Thermal Monitor Table<br />
</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Overclocking Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to Overclocking Table<br />
</p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>LowPower NVLINK Table Pointer</p></td>
<td style="text-align: right;"><p>32</p></td>
<td style="text-align: left;"><p>Pointer to LPWR NVLINK Table<br />
</p></td>
</tr>
</tbody>
</table>

</div>

<div class="paragraph">

Note 1: Notes on the **Performance Settings Script Pointer**:

</div>

<div class="ulist">

  - Intended to be parsed by system software to directly obtain register
    settings for a particular P-state, typically P0.

  - May be called as a subscript from the primary device initialization
    script so that it can be used both for initialization and to provide
    data for P-state changes.

  - Must not contain any conditions or opcodes that require reading
    hardware, with the exception of INIT\_XMEMSEL\* opcodes that only
    need to read the memory strap, which may be cached by system
    software.

  - System software is only required to parse up to the first occurrence
    of the desired register.

  - Parsing should terminate at the INIT\_DONE opcode.  

</div>

### BIT\_STRING\_PTRS (<span style="color: red;">Version 1</span>)

<div style="clear:left">

</div>

<div class="paragraph">

**<span style="color: red;">This data structure has been
deprecated.</span>**

</div>

<div class="paragraph">

This data structure contains pointers to strings in the VBIOS image

</div>

<div class="ulist">

  - All of the strings in this structure are ‘0’ terminated.

  - The “Size” bytes indicate the maximum length available for storing
    the string, non-inclusive of the terminating 0.

</div>

<div class="tableblock">

| Name                           | Bit width | Values and Meaning                                                                                                                                    |
| :----------------------------- | --------: | :---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sign On Message Poitner        |        16 | Pointer to Sign On Message                                                                                                                            |
| Sign On Message Maximum Length |         8 | Maximum length of Sign On Message                                                                                                                     |
| OEM String                     |        16 | OEM String to identify graphics controller chip or product family. This is the last radix in the combined version string, e.g. 25 in *70.18.01.00.25* |
| OEM String Size                |         8 | Maximum length of OEM string                                                                                                                          |
| OEM Vendor Name                |        16 | Name of the vendor that produced the display controller board product                                                                                 |
| OEM Vendor Name Size           |         8 | Maximum length of OEM Vendor Name                                                                                                                     |
| OEM Product Name               |        16 | Product name of the controller board                                                                                                                  |
| OEM Product Name Size          |         8 | Maximum length of OEM Product Name                                                                                                                    |
| OEM Product Revision           |        16 | Revision of manufacturing level of the display controller board                                                                                       |
| OEM Product Revision Size      |         8 | Maximum length of OEM Product Revision                                                                                                                |

</div>

### BIT\_STRING\_PTRS (Version 2)

<div style="clear:left">

</div>

<div class="paragraph">

This data structure contains pointers to strings in the VBIOS image

</div>

<div class="ulist">

  - All of the strings in this structure are ‘0’ terminated.

  - The “Size” bytes indicate the maximum length available for storing
    the string, non-inclusive of the terminating 0.

</div>

<div class="tableblock">

| Name                           | Bit width | Values and Meaning                                                                                                                                    |
| :----------------------------- | --------: | :---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sign On Message Poitner        |        16 | Pointer to Sign On Message                                                                                                                            |
| Sign On Message Maximum Length |         8 | Maximum length of Sign On Message                                                                                                                     |
| Version String                 |        16 | Pointer to the "Version ww.xx.yy.zz" string                                                                                                           |
| Version String Size            |         8 | Maximum length of the version string                                                                                                                  |
| Copyright String               |        16 | Pointer to the copyright string                                                                                                                       |
| Copyright String Size          |         8 | Maximum length of the copyright string                                                                                                                |
| OEM String                     |        16 | OEM String to identify graphics controller chip or product family. This is the last radix in the combined version string, e.g. 25 in *70.18.01.00.25* |
| OEM String Size                |         8 | Maximum length of OEM string                                                                                                                          |
| OEM Vendor Name                |        16 | Name of the vendor that produced the display controller board product                                                                                 |
| OEM Vendor Name Size           |         8 | Maximum length of OEM Vendor Name                                                                                                                     |
| OEM Product Name               |        16 | Product name of the controller board                                                                                                                  |
| OEM Product Name Size          |         8 | Maximum length of OEM Product Name                                                                                                                    |
| OEM Product Revision           |        16 | Revision of manufacturing level of the display controller board                                                                                       |
| OEM Product Revision Size      |         8 | Maximum length of OEM Product Revision                                                                                                                |

</div>

### BIT\_TMDS\_PTRS

<div style="clear:left">

</div>

<div class="paragraph">

This data structure contains TMDS Initialization table pointers.

</div>

<div class="tableblock">

| Name                    | Bit width | Values and Meaning         |
| :---------------------- | --------: | :------------------------- |
| TMDS Info Table Pointer |        16 | Pointer to TMDS Info Table |

</div>

### BIT\_DISPLAY\_PTRS

<div style="clear:left">

</div>

<div class="paragraph">

This data structure contains Display Control/Programming related
pointers.

</div>

<div class="tableblock">

| Name                            | Bit width | Values and Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| :------------------------------ | --------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Display Scripting Table Pointer |        16 | Pointer to Display Scripting Table                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Display Control Flags           |         8 | Display Control Flags byte : \[0:0\] = Enable white overscan border for diagnostic purposes : \[1:1\] = NO\_DISPLAY\_SUBSYSTEM: Display subsystem isn’t included in the GPU (used for displayless coproc) : \[2:2\] = DISPLAY\_FPGA: Display subsystem is on an FPGA (used for pre-SI testing). : \[3:3\] = VBIOS avoids touching mempool while drivers running : \[4:4\] = Offset PCLK between 2 heads : \[5:5\] = Boot with DP Hotplug disabled : \[6:6\] = Allow detection of DP sinks by doing a DPCD register read : \[7:7\] = Reserved |
| SLI Table Header Pointer        |        16 | Pointer to the SLI Table Header                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |

</div>

### BIT\_VIRTUAL\_PTRS

<div style="clear:left">

</div>

<div class="paragraph">

This data structure contains Virtual Field pointers.

</div>

<div class="tableblock">

| Name                              | Bit width | Values and Meaning                                           |
| :-------------------------------- | --------: | :----------------------------------------------------------- |
| Virtual Strap Field Table Pointer |        16 | Pointer to Virtual Strap Field Table                         |
| Virtual Strap Field Register      |        16 | Virtual STrap Field Register                                 |
| Translation Table Pointer         |        16 | Pointer to translation table so virtual straps can be sparse |

</div>

### BIT\_32BIT\_PTRS

<div style="clear:left">

</div>

<div class="paragraph">

This data structure contains BIOS related data that is located outside
the 64K ROM image.

</div>

<div class="ulist">

  - It is used by the VBIOS to access tables during POST that need to be
    copied into the runtime image.

  - No data structure is currently defined.

</div>

### BIT\_DP\_PTRS

<div style="clear:left">

</div>

<div class="paragraph">

This data structure contains the DP Info Table.

</div>

<div class="tableblock">

| Name                  | Bit width | Values and Meaning       |
| :-------------------- | --------: | :----------------------- |
| DP Info Table Pointer |        16 | Pointer to DP Info Table |

</div>

### BIT\_PMU\_PTRS (<span style="color: red;">Version 1</span>)

<div style="clear:left">

</div>

<div class="paragraph">

<span style="color: red;">This data structure has been deprecated. It is
superseded by BIT\_FALCON\_DATA (Version 2).</span>

</div>

<div class="paragraph">

This data structure contains PMU-related pointers

</div>

<div class="tableblock">

| Name                                  | Bit width | Values and Meaning                                                                                       |
| :------------------------------------ | --------: | :------------------------------------------------------------------------------------------------------- |
| PMU Function Table Pointer            |        16 | Pointer to PMU Function Table (Deprecated)                                                               |
| PMU Function Table Pointer (32-bit)   |        32 | Pointer to PMU Function Table (32-bit)                                                                   |
| PMU Init-From-Rom Code Image Pointer  |        32 | core80-: Pointer to PMU IFR code image in Kepler format core82+: Pointer to IFR IMEM image in raw format |
| PMU Init-From-Rom Code Image Size     |        32 | core80-: Size of PMU IFR code image in Kepler format core82+: Size of IFR IMEM image in raw format       |
| PMU Init-From-Rom Code Image ID       |         8 | ID of PMU IFR code image                                                                                 |
| PMU Init-From-Rom Code Image Info Ptr |        32 | Pointer to info struct for IFR code image                                                                |
| PMU Init-From-Rom Data Image Ptr      |        32 | core82+: Pointer to IFR DMEM image in raw format                                                         |
| PMU Init-From-Rom Data Image Size     |        32 | core82+: Size of IFR DMEM image in raw format                                                            |

</div>

### BIT\_FALCON\_DATA (Version 2)

<div style="clear:left">

</div>

<div class="paragraph">

This data structure contains Falcon-related data and pointers. It
supersedes [BIT\_PMU\_PTRS (Version 1)](#BIT_PMU_PTRS_v1). The name was
changed for version 2 to better reflect the scope of associated data.

</div>

<div class="tableblock">

| Name                       | Bit width | Values and Meaning            |
| :------------------------- | --------: | :---------------------------- |
| Falcon Ucode Table Pointer |        32 | Pointer to Falcon Ucode Table |

</div>

### BIT\_UEFI\_DATA

<div style="clear:left">

</div>

<div class="paragraph">

This data structure contains the UEFI Driver Data structure

</div>

<div class="tableblock">

| Name                        | Bit width | Values and Meaning                                                                                                                                                                                                                          |
| :-------------------------- | --------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Minimum UEFI Driver Version |        32 | Minimum UEFI driver version that is required when merging with the legacy VBIOS image                                                                                                                                                       |
| UEFI Compatibility Level    |         8 | Specifies the legacy VBIOS UEFI compatibility level which can be used to prevent the legacy VBIOS from being merged with an incompatible UEFI driver                                                                                        |
| UEFI Flags                  |        64 | UEFI Flags : \[0:0\] Display switch support :: 0 = Enabled :: 1 = Disabled : \[1:1\] LCD diagnostics support :: 0 = Disabled :: 1 = Enabled : \[2:2\] Glitchless support :: 0 = Enabled :: 1 = Disabled : \[63:3\] Reserved (defaults to 0) |

</div>

### BIT\_MXM\_DATA

<div style="clear:left">

</div>

<div class="paragraph">

This data structure contains the MXM Configuration Data structure

</div>

<div class="tableblock">

| Name                                | Bit width | Values and Meaning                                                                                                                                                                                                                                                                                                                                                                                                                           |
| :---------------------------------- | --------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Module Spec Version                 |         8 | The BCD version of the Thermal Electromechanical Specification the module was designed for                                                                                                                                                                                                                                                                                                                                                   |
| Module Flags 0                      |         8 | Module Flags 0 byte : \[3:0\] - Form Factor :: 0x0 = Not MXM :: 0x1 = Type-I :: 0x2 = Type-II :: 0x3 = Type-III :: 0x4 = Type-IV :: 0x5-0xE = Reserved :: 0xF = Undefined : \[7:4\] - Reserved                                                                                                                                                                                                                                               |
| Config Flags 0                      |         8 | Configuration Flags 0 byte : \[0:0\] = MXM Structure Required : \[1:1\] = MXM Structure validation failed : \[3:2\] = DCB modification status :: 0 = VBIOS modification complete :: 1-2 = Reserved :: 3 = MXM Default DCB : \[7:4\] – Chip package type of GPU on the MXM module :: 0x0 = Package older than G3 type :: 0x1 = G3 package :: 0x2 = GB1-128/256 package :: 0x3 = GB1-64 package :: 0x4 = GB4-256 package :: 0x5-0xF = Reserved |
| DP Drive Strength Scale             |         8 | Used to modify the DP Drive Strength for DP in MXM30                                                                                                                                                                                                                                                                                                                                                                                         |
| MXM Digital Connector Table Pointer |        16 | Pointer to table for mapping MXM Digital Connection number into SOR/Sublinks config                                                                                                                                                                                                                                                                                                                                                          |
| MXM DDC/Aux to CCB Table Pointer    |        16 | Pointer to table for mapping MXM DDC/Aux number CCB port number                                                                                                                                                                                                                                                                                                                                                                              |

</div>

### BIT\_BRIDGE\_FW\_DATA

<div style="clear:left">

</div>

<div class="paragraph">

This data structure contains the Bridge Firmware Data structure

</div>

<div class="tableblock">

| Name                          | Bit width | Values and Meaning                                                                                                                            |
| :---------------------------- | --------: | :-------------------------------------------------------------------------------------------------------------------------------------------- |
| Firmare Version               |        32 | Firmware Binary Version                                                                                                                       |
| Firmware OEM Version          |         8 | Firmware OEM Verison Number                                                                                                                   |
| Firmware Image Length         |        16 | Firmware Image Length in increments of 512 bytes                                                                                              |
| BIOSMOD Date                  |        64 | Date of Last BIOSMod Modification                                                                                                             |
| Firmware Flags                |        32 | Firmware Flags : \[0:0\] Build :: 0 = Release :: 1 = Engineering : \[1:1\] I2C :: 0 = Master (possible I2C slave connected) :: 1 = Not Master |
| Engineering Product Name      |        16 | Pointer to the Engineering Product Name                                                                                                       |
| Engineering Product Name Size |         8 | Maximum length of the Engineering Product Name string                                                                                         |

</div>

</div>

<div id="footer">

<div id="footer-text">

Last updated 2018-01-26 11:56:33 PDT

</div>

</div>
