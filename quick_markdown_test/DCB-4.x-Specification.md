<div id="header">

# Device Control Block 4.x Specification

</div>

<div id="content">

<div class="sect1">

## Purpose

<div class="sectionbody">

<div class="paragraph">

Device Control Blocks (DCBs) are static tables used to describe the
board topology and connections external to the GPU chip.

</div>

<div class="paragraph">

Each board built has specific additions to the capabilities through
external devices, as well as limitations where output lines are not
linked to device connectors. DCBs define the devices connected, specific
information needed to configure those devices, and the external
electrical connections such as HDMI and Display Port.

</div>

<div class="paragraph">

DCBs do not try to explain the capabilities of the chip itself. That
information is implicit in the VBIOS, firmware and drivers, which are
built differently for each chip. Both the firmware and the drivers know
the inherent capability of each chip, and use runtime choices to
determine chip dependent code paths.

</div>

<div class="ulist">

<div class="title">

DCB version and use

</div>

  - DCB 1.x is used with Core3 VBIOS (NV5, NV10, NV11, NV15, NV20).

  - DCB 2.x (2.0-2.4) is used with Core4 and Core4r2 VBIOS (NV17, NV25,
    NV28, NV3x).

  - DCB 3.0 is used with Core5 VBIOS (NV4x, G7x).

  - DCB 4.x is used with Core6, Core7, and Core8 VBIOS (G80+).

</div>

<div class="sect2">

### DCB 4.1 Changes

<div class="paragraph">

With the GM20x family of chips, any SOR could be used to drive any
analog pad link on the GPU. Using SORs and sublinks as fixed constants
for a DCB device entry no longer accurately described the board
topology. To fix this, DCB 4.1 repurposes the Output Resource Assignment
Mask and the Sublink Assignment Mask as a Pad Macro Assignment mask and
a Pad Link Assignment mask respectively for Digital Flat Panel Device
Entries.

</div>

</div>

</div>

</div>

<div class="sect1">

## Device Control Block Structure

<div class="sectionbody">

<div class="paragraph">

The 4.x DCB Data Structure consists of the following parts:

</div>

<div class="ulist">

  - Header - The version number (e.g., 0x40 for Version 4.0), the header
    size, the size of each DCB Entry (currently 8 bytes), the number of
    valid DCB Entries, pointers to different tables, and the DCB
    signature. If any of the pointers here are NULL, then those tables
    are considered to be absent or invalid.

  - Device entries list - One for each display connector (two for DVI-I
    connectors). Each device entry is subdivided into two main parts:
    [Display Path Information](#_display_path_information) and [Device
    Specific Information](#_device_specific_information).

</div>

</div>

</div>

<div class="sect1">

## Device Control Block Header

<div class="sectionbody">

<div class="tableblock">

| Name                                 | Bit width | Optional/Mandatory | Values and Meaning                                                                                                                                                      |
| :----------------------------------- | --------: | :----------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Version                              |         8 |         O          | Version \# of the DCB Header and Entries. E.g., DCB 4.0 will start with a value of 0x40 here. A version number of zero directs the driver to use an internal DCB table. |
| Header Size                          |         8 |         M          | Size of the DCB Header in Bytes. For v4.0 this will be 27 bytes.                                                                                                        |
| Entry Count                          |         8 |         M          | Number of [DCB Device Entries](#_dcb_device_entries) immediately following this table.                                                                                  |
| Entry Size                           |         8 |         M          | Size of Each Entry in bytes. With the start of DCB 4.0, this field should be 8.                                                                                         |
| Communications Control Block Pointer |        16 |         M          | Pointer to the [Communications Control Block](#_communications_control_block). In v3.0 this was the I2C Control Block Pointer.                                          |
| DCB Signature                        |        32 |         M          | DCB signature = 0x4EDCBDCB. This is used to tell a valid DCB from an invalid one.                                                                                       |
| GPIO Assignment Table Pointer        |        16 |         M          | Pointer to the [GPIO Assignment Table](#_gpio_assignment_table).                                                                                                        |
| Input Devices Table Pointer          |        16 |         O          | Pointer to the [Input Devices Table](#_input_devices_table).                                                                                                            |
| Personal Cinema Table Pointer        |        16 |         O          | Pointer to the [Personal Cinema Table](#_personal_cinema_table).                                                                                                        |
| Spread Spectrum Table Pointer        |        16 |         O          | Pointer to the [Spread Spectrum Table](#_spread_spectrum_table).                                                                                                        |
| I2C Devices Table Pointer            |        16 |         O          | Pointer to the [I2C Devices Table](#_i2c_device_table).                                                                                                                 |
| Connector Table Pointer              |        16 |         M          | Pointer to the [Connector Table](#_connector_table).                                                                                                                    |
| Flags                                |         8 |         M          | See [DCB Flags](#_dcb_flags) below                                                                                                                                      |
| HDTV Translation Table Pointer       |        16 |         O          | Pointer to the [HDTV Translation Table](#_hdtv_translation_table). This structure is optional. If the structure is not needed, then this pointer can be set to 0.       |
| Switched Outputs Table Pointer       |        16 |         O          | Pointer to the [Switched Outputs Table](#_switched_outputs_table). This structure is optional.                                                                          |

Table 1. Device Control Block Header

</div>

<div class="paragraph">

An "optional" table pointer or field may be set to zero to indicate that
no table is present. If the structure is not needed, then this pointer
can be set to 0.

</div>

<div class="admonitionblock">

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td><div class="title">
Note
</div></td>
<td>Throughout this document, a "pointer" means a byte offset relative to the start of the VBIOS image.</td>
</tr>
</tbody>
</table>

</div>

<div class="sect2">

### DCB Flags

<div class="paragraph">

Each bit flag has a different meaning. All undefined bits are reserved
and must be set to 0.

</div>

<div class="ulist">

<div class="title">

DCB Flag Bits

</div>

  - Bit 0 - Boot Display Count:
    
    <div class="ulist">
    
      - 0 - Only 1 boot display is allowed.
    
      - 1 - 2 boot displays are allowed.
    
    </div>

</div>

<div class="paragraph">

These next 2 bits are all used for VIP connections.

</div>

<div class="ulist">

  - Bits 5:4 - VIP location. Possible values are:
    
    <div class="ulist">
    
      - 00b - No VIP.
    
      - 01b - VIP is on Pin Set A.
    
      - 10b - VIP is on Pin Set B.
    
      - 11b - Reserved
    
    </div>

</div>

<div class="paragraph">

These next 2 bits are used for Distributed Rendering (DR) configuration.

</div>

<div class="ulist">

  - Bit 6 - All capable DR ports: Pin Set A:
    
    <div class="ulist">
    
      - 1 - Pin Set A is routed to a SLI Finger.
    
      - 0 - Pin Set A is not attached.
    
    </div>

  - Bit 7 - All capable DR Ports: Pin Set B:
    
    <div class="ulist">
    
      - 1 - Pin Set B is routed to a SLI Finger.
    
      - 0 - Pin Set B is not attached.
    
    </div>

</div>

<div class="admonitionblock">

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td><div class="title">
Note
</div></td>
<td>A PIOR port cannot be used both as a Distributed Rendering connection and as an Output Display at the same time.</td>
</tr>
</tbody>
</table>

</div>

</div>

<div class="sect2">

### DCB Header Version 4.x Sizes

<div class="paragraph">

The v4.x DCB header has added fields over time.

</div>

<div class="tableblock">

|   DATE   | New Size |          Last Inclusive Field           |
| :------: | :------: | :-------------------------------------: |
|  Start   | 23 Bytes |                  Flags                  |
| 08-02-06 | 25 Bytes | DCB 3.0, HDTV Translation Table Pointer |
| 11-07-06 | 27 Bytes |     Switched Outputs Table Pointer      |

</div>

</div>

</div>

</div>

<div class="sect1">

## DCB Device Entries

<div class="sectionbody">

<div class="paragraph">

A DCB device entry is 64 bits wide, two double words. The first 32 bits,
[Display Path Information](#_display_path_information), contain the main
routing information. Their format is common to all devices. The second
32 bits, [Device Specific Information](#_device_specific_information),
are interpreted based on the Type field from the Display Path
Information.

</div>

<div class="paragraph">

There is one device entry for each output display path. The number of
DCB entries is listed in the DCB Header.

</div>

<div class="admonitionblock">

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td><div class="title">
Note
</div></td>
<td>For DVI-I connectors there are two entries: one for the CRT and one for the LCD. The two device entries share the same I2C port.</td>
</tr>
</tbody>
</table>

</div>

<div class="paragraph">

Device Entries are listed in order of boot priority. The VBIOS code will
iterate through the DCB entries and if a device is found, then that
device will be configured. If not, the VBIOS moves to the next index in
the DCB. If no device is found, the first CRT on the list should be
chosen.

</div>

<div class="paragraph">

GPUs earlier than G80 have a "mirror mode" feature that enables up to
two display devices to be enabled by the VBIOS, and controlled through
the same VGA registers. G80 and later display hardware only supports one
display in VGA mode, and the VBIOS will only enable one display device.

</div>

<div class="paragraph">

When Device Entries are listed, it is not allowed to have two entries
for the same output device.

</div>

<div class="sect2">

### Display Path Information

<div class="tableblock">

31

</div>

</div>

</div>

</div>

</div>

30

29

28

27

26

25

24

23

22

21

20

19

18

17

16

15

14

13

12

11

10

9

8

7

6

5

4

3

2

1

0

Rsvd

VD

Output Devices

BBDR

BDR

Loc

Bus

Connector

Head

EDID Port

Type

<div class="tableblock">

| Name | Bit width | Values and Meaning                                               |
| :--: | :-------- | :--------------------------------------------------------------- |
| Type | 4         | Display type.                                                    |
| EDID | 4         | EDID Port.                                                       |
| Head | 4         | Head bitmask.                                                    |
| Con  | 4         | Connector table entry index.                                     |
| Bus  | 4         | Logical bus, used for mutual exclusion.                          |
| Loc  | 2         | Location of the final stage devices, on-chip or off-chip.        |
| BDR  | 1         | Disables this as a boot display if set.                          |
| BBDR | 1         | If set, disables the ability to boot if not display is detected. |
|  VD  | 1         | Indicates this is a virtual device.                              |
| Rsvd | 3         | Reserved, set to 0.                                              |

</div>

<div class="paragraph">

<div class="title">

Type

</div>

This field defines the Type of the display used on this display path.
Currently defined values are:

</div>

<div class="ulist">

  - 0 = CRT

  - 1 = TV

  - 2 = TMDS

  - 3 = LVDS

  - 4 = Reserved

  - 5 = SDI

  - 6 = DisplayPort

  - 8 = Reserved

  - E = EOL (End of Line) - This signals the SW to stop parsing any more
    entries.

  - F = Skip Entry - This allows quick removal of entries from DCB.

</div>

<div class="paragraph">

Note: LVDS entries must precede eDP entries to meet RM requirements and
avoid glitches during detection.

</div>

<div class="paragraph">

<div class="title">

EDID Port

</div>

Each number refers to an entry in the [Communications Control
Block](#_communications_control_block) Structure that represents the
port to use in order to query the EDID. This number cannot be equal to
or greater than the Communication Control Block Header’s Entry Count
value, except if the EDID is not retrieved via DDC (over I2C or DPAux).

</div>

<div class="paragraph">

For DFPs, if the EDID source is set to straps or SBIOS, then this field
must be set to 0xF to indicate that we are not using a Communications
Control Block port for this device to get the EDID.

</div>

<div class="paragraph">

<div class="title">

Head Bitmask

</div>

Each bit defines the ability of that head with this device.

</div>

<div class="ulist">

  - Bit 0 = Head 0

  - Bit 1 = Head 1

  - Bit 2 = Head 2

  - Bit 3 = Head 3

</div>

<div class="paragraph">

GPUs before GK107 only support two heads. For those devices, bits 2 and
3 should always be zero.

</div>

<div class="paragraph">

<div class="title">

Connector Index

</div>

This field signifies a specific entry in the Connector Table. More than
one DCB device can have the same Connector Index. This number cannot be
equal to or greater than the Connector Table Header’s Entry Count value.

</div>

<div class="admonitionblock">

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td><div class="title">
Note
</div></td>
<td>If two DCB entries have the same Connector Index, that still allows them to be displayed at the same time. To prevent combinations based on the connector, use the Bus field.</td>
</tr>
</tbody>
</table>

</div>

<div class="paragraph">

<div class="title">

Bus

</div>

This field only allows for logical mutual exclusion of devices so that
they cannot display simultaneously. The driver uses this field to
disallow the use of a combination of two devices if they share the same
bus number.

</div>

<div class="paragraph">

<div class="title">

Location

</div>

This field shows the location of the last output device before the data
is sent off from our board to the display.

</div>

<div class="paragraph">

Currently defined values are:

</div>

<div class="ulist">

  - 0 = On Chip (internal) TV encoder, internal TMDS encoder

  - 1 = On Board (external) DAC, external TMDS encoder

  - 2 = Reserved.

</div>

<div class="ulist">

<div class="title">

Boot Device Removed

</div>

  - 0 = This device is allowed to boot if detected.

  - 1 = This device is not allowed to boot, even if detected.

</div>

<div class="ulist">

<div class="title">

Blind Boot Device Removed

</div>

  - 0 = This device is allowed to boot if no devices are detected.

  - 1 = This device is not allowed to boot if no devices are detected.

</div>

<div class="paragraph">

<div class="title">

DAC/SOR/PIOR Assignment (Output Resource) for DCB 4.x:

</div>

Each bit defines the use of this connector with a DAC for internal CRTs
and TVs, an SOR for internal DFPs, and a PIOR for external devices like
TMDS, SDI or TV Encoders.

</div>

<div class="paragraph">

Currently defined values are:

</div>

<div class="ulist">

  - Bit 0 = DAC 0, SOR 0, or PIOR 0

  - Bit 1 = DAC 1, SOR 1, or PIOR 1

  - Bit 2 = DAC 2, SOR 2, or PIOR 2

  - Bit 3 = DAC 3, SOR 3, or PIOR 3

</div>

<div class="paragraph">

<div class="title">

DAC/SOR/PIOR/Pad Macro Assignment (Output Resource) for DCB 4.1:

</div>

For CRT or External Encoder Device Entries:

</div>

<div class="paragraph">

Each bit defines the use of this connector with a DAC for internal CRTs
and TVs, and a PIOR for external devices like TMDS, SDI or TV Encoders.

</div>

<div class="paragraph">

Currently defined values are:

</div>

<div class="ulist">

  - Bit 0 = DAC 0 or PIOR 0

  - Bit 1 = DAC 1 or PIOR 1

  - Bit 2 = DAC 2 or PIOR 2

  - Bit 3 = DAC 3 or PIOR 3

</div>

<div class="paragraph">

For Internal or External DFP Device Entries:

</div>

<div class="paragraph">

For Internal and External DFPs we can use any of the available SORs but
the Pad Macro is fixed per Device Entry.

</div>

<div class="ulist">

  - Bit 0 = Pad Macro 0 (Pad Links A and B)

  - Bit 1 = Pad Macro 1 (Pad Links C and D)

  - Bit 2 = Pad Macro 2 (Pad Links E and F)

  - Bit 3 = Pad Macro 3 (Pad Link G)

</div>

<div class="ulist">

<div class="title">

Virtual Device

</div>

  - 0 = This is a physical device.

  - 1 = This is a virtual device.

</div>

<div class="paragraph">

Virtual devices are used only for remote desktop rendering. When set to
1, EDID Port should be set to 0xF (unused) and the Connector Index
should reference an entry with Type="Skip Entry".

</div>

<div class="paragraph">

<div class="title">

Extra Information

</div>

The BUS field may reflect only a logical limitation of the buses. It can
describe an actual physical limitation, or it may be solely a way to
remove the combination between two DCB entries.

</div>

<div class="sect2">

### Device Specific Information

<div class="paragraph">

Each device type has a different specific information associated with
it. However, TMDS, LVDS, SDI, and DisplayPort share the same DFP
Specific Information.

</div>

<div class="sect3">

#### CRT Specific Information

<div class="tableblock">

| 31..0               |
| :------------------ |
| Reserved (Set to 0) |

</div>

</div>

<div class="sect3">

#### DFP Specific Information

<div class="paragraph">

DFP Specific Information is used to decribe TMDS, LVDS, SDI and
DisplayPort Types of devices.

</div>

<div class="tableblock">

| 31..28 | 27..24 | 23..21 | 20 | 19..18 | 17   | 16   | 15..8   | 7..6 | 5..4   | 3..2 | 1..0 |
| :----- | :----- | :----- | :- | :----- | :--- | :--- | :------ | :--- | :----- | :--- | :--- |
| Rsvd   | MxLM   | MLR    | E  | Rsvd   | HDMI | Rsvd | Ext Enc | Rsvd | SL/DPL | Ctrl | EDID |

</div>

<div class="tableblock">

|  Name   | Bit width | Values and Meaning                            |
| :-----: | :-------: | :-------------------------------------------- |
|  EDID   |     2     | EDID source.                                  |
|  Ctrl   |     2     | Power and Backlight Control.                  |
| SL/DPL  |     2     | Sub-link/DisplayPort Link/Pad Link Assignment |
|  Rsvd   |     2     | Reserved, set to 0.                           |
| Ext Enc |     8     | External Link Type.                           |
|  Rsvd   |     1     | Reserved.                                     |
|  HDMI   |     1     | HDMI Enable.                                  |
|  Rsvd   |     2     | Reserved, set to 0.                           |
|    E    |     1     | External Communications Port.                 |
|  MxLR   |     3     | Maximum Link Rate.                            |
|  MxLM   |     4     | Maximum Lane Mask.                            |
|  Rsvd   |     4     | Reserved, set to 0.                           |

</div>

<div class="paragraph">

<div class="title">

EDID source

</div>

This field states where to get the EDIDs for the panels. Current values
are:

</div>

<div class="ulist">

  - 0 = EDID is read via DDC.

  - 1 = EDID is determined via Panel Straps and VBIOS tables.

  - 2 = EDID is obtained using the \_DDC ACPI interface or VBIOS 5F80/02
    SBIOS INT15 calls.

  - 3 = Reserved.

</div>

<div class="paragraph">

<div class="title">

Mobile LVDS Detection Policy

</div>

There is a secondary fallback policy that is used for all mobile LVDS
panels. It follows this convention:

</div>

<div class="tableblock">

DCB EDID Source

</div>

</div>

</div>

Panel Strap

Panel Index

EDID Retrieval

DDC

\== 0xF

Don’t Care.

Use DDC.

\!= 0xF

Don’t Care.

Use Straps and VBIOS Tables.

Straps and VBIOS Tables

\!= 0xF

Don’t Care.

Use Straps and VBIOS Tables.

\== 0xF

\!= 0xF

Use Straps and VBIOS Tables.

\== 0xF

No Panel.

SBIOS

Don’t Care.

Don’t Care.

Use SBIOS \_DDC ACPI method or SBIOS/VBIOS Call.

<div class="paragraph">

If the board designer chooses to use DDC based EDIDs always, the VBIOS
can override the Panel Strap to always indicate 0xF via SW Strap
Overrides or through the DevInit scripts.

</div>

<div class="paragraph">

<div class="title">

Power and Backlight Control

</div>

This field describes the control method for the power and backlight of
the panel. Currently defined values are:

</div>

<div class="ulist">

  - 0 = External. This is used to define panels where we don’t have
    direct control over the power or backlight. For example, this value
    is used for most TMDS panels.

  - 1 = Scripts. Used for most LVDS panels.

  - 2 = VBIOS callbacks to the SBIOS.

</div>

<div class="paragraph">

<div class="title">

Sub-link/DisplayPort Link/Pad Link Assignment

</div>

This field specifies a board-supported sub-link mask for TMDS, LVDS, and
SDI. For Display Port, this field specifies the link mask supported on
the board.

</div>

<div class="paragraph">

For DCB 4.x: For TMDS, LVDS, and SDI, this field lists which sub-links
in each SOR are routed to the connector on the board.

</div>

<div class="paragraph">

Possible sub-link values are:

</div>

<div class="ulist">

  - Bit 0: Sub-link A

  - Bit 1: Sub-link B

</div>

<div class="paragraph">

If both sub-links are routed to the connector, specifying a dual-link
connector, then bits 0 and 1 will both be set.

</div>

<div class="admonitionblock">

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td><div class="title">
Note
</div></td>
<td>Dual-link hook-up does not necessarily mean that both links should be used during programming. According to the DVI 1.0 specification, the crossover frequency of 165 MHZ should be the deciding factor for when dual-link vs. single-link connections should be used for TMDS use. This field merely indicates whether the connector has two links connected to it. It does not specify the actual use of either single-link or dual-link connections.</td>
</tr>
</tbody>
</table>

</div>

<div class="paragraph">

LVDS uses single-link or dual-link connections based on the individual
panel model’s requirements. For example, SXGA panels may be run with
single-link or dual-link LVDS connections.

</div>

<div class="paragraph">

For DisplayPort, this field describes which links in each SOR are routed
to the connector on the board. Possible link values are:

</div>

<div class="ulist">

  - Bit 0: DP-A (Display Port Resource A)

  - Bit 1: DP-B (Display Port Resource B)

</div>

<div class="admonitionblock">

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td><div class="title">
Note
</div></td>
<td>Unlike TMDS, LVDS, and SDI, if both links are routed to the connector, this does not indicate the presence of a dual-link connector. It simply means that both Display Port (DP) resources A and B may be used with this SOR. That is: DP-A or DP-B may be associated with an output device (OD) to output via DisplayPort, but not both simultaneously.</td>
</tr>
</tbody>
</table>

</div>

<div class="paragraph">

For DCB 4.1: For TMDS/LVDS/DP this field describes which links of the
Pad Macro are routed to the connector on the board.

</div>

<div class="ulist">

  - Bit 0: Pad Link 0

  - Bit 1: Pad Link 1

</div>

<div class="paragraph">

<div class="title">

Reserved

</div>

Set to 0.

</div>

<div class="paragraph">

<div class="title">

External Link Type

</div>

This field describes the exact external link used on the board. If this
Location field in the Display Path of this DCB entry is set to ON CHIP,
then these bits should be set to 0.

</div>

<div class="paragraph">

Currently defined values:

</div>

<div class="paragraph">

<div class="title">

HDMI Enable

</div>

This bit is placed here to allow the use of HDMI on this particular DFP
output display.

</div>

<div class="paragraph">

Currently defined values are:

</div>

<div class="ulist">

  - 0 = Disable HDMI on this DFP

  - 1 = Enable HDMI on this DFP

</div>

<div class="paragraph">

<div class="title">

External Communications Port

</div>

If this device uses external I2C or DPAux communication, then this field
allows us to know which port is to be used. If the device is internal to
the chip, set this bit to 0 by default.

</div>

<div class="paragraph">

Currently defined values are:

</div>

<div class="ulist">

  - 0 = Primary Communications Port

  - 1 = Secondary Communications Port

</div>

<div class="paragraph">

The [Communications Control Block
Header](#_communications_control_block_header) holds the primary and
secondary port indices. Each index maps to an entry in the
[Communications control Block](#_communications_control_block) table,
which specifies the physical port and type to use to communicate with
this device.

</div>

<div class="paragraph">

<div class="title">

Maximum Link Rate

</div>

This field describes the maximum link rate allowed for the links within
the Display Port connection. This field is only applicable to
DisplayPort device types.

</div>

<div class="paragraph">

Possible values are:

</div>

<div class="tableblock">

|   |           |
| :- | :-------- |
| 0 | 1.62 Gbps |
| 1 | 2.7 Gbps  |
| 2 | 5.4 Gbps  |
| 3 | 8.1 Gbps  |

</div>

<div class="paragraph">

<div class="title">

Maximum Lane Count

</div>

This field describes the maximum lanes that are populated on the board.
This field is only applicable to DisplayPort device types.

</div>

<div class="paragraph">

Possible values are:

</div>

<div class="tableblock">

|     |                                                                         |
| :-- | :---------------------------------------------------------------------- |
| 0x1 | 1 Lane                                                                  |
| 0x2 | 2 Lanes --- This value will be applicable only on Maxwell & Later chips |
| 0x3 | 2 Lanes --- deprecated, will be removed in DCB 6.0                      |
| 0x4 | 4 Lanes --- This value will be applicable only on Maxwell & Later chips |
| 0xF | 4 Lanes --- deprecated, will be removed in DCB 6.0                      |

</div>

<div class="tableblock">

| Value | Name                                                                                | I2C Addr                            |
| :---- | :---------------------------------------------------------------------------------- | :---------------------------------- |
| 0     | Undefined (allows backward compatibility) - Assumes Single-Link.                    |                                     |
| 1     | Silicon Image 164 - Single-Link TMDS.                                               | 0x70                                |
| 2     | Silicon Image 178 - Single-Link TMDS.                                               | 0x70                                |
| 3     | Dual Silicon Image 178 - Dual-Link TMDS.                                            | 0x70 (primary), 0x72 (secondary)    |
| 4     | Chrontel 7009 - Single-Link TMDS.                                                   | 0xEA                                |
| 5     | Chrontel 7019 - Dual-Link LVDS.                                                     | 0xEA                                |
| 6     | National Semiconductor DS90C387 - Dual Link LVDS.                                   |                                     |
| 7     | Silicon Image 164 - Single-Link TMDS (Alternate Address).                           | 0x74                                |
| 8     | Chrontel 7301 - Single-Link TMDS.                                                   |                                     |
| 9     | Silicon Image 1162 - Single Link TMDS (Alternate Address).                          | 0x72                                |
| A     | Reserved                                                                            | Reserved                            |
| B     | Analogix ANX9801 - 4-Lane DisplayPort (deprecated on Fermi+).                       | 0x70 (transmitter), 0x72 (receiver) |
| C     | Parade Tech DP501 - 4-Lane DisplayPort.                                             |                                     |
| D     | Analogix ANX9805 - HDMI and DisplayPort (deprecated on Fermi+).                     | 0x70, 0x72, 0x7A, 0x74              |
| E     | Analogix ANX9805 - HDMI and DisplayPort (Alternate Address) (deprecated on Fermi+). | 0x78, 0x76, 0x7E, 0x7C              |

Table 2. External Link Type

</div>

<div class="sect3">

#### TV Specific Information

<div class="tableblock">

| 31..24 | 23..20 | 19..16 | 15..8 | 7..4  | 3..0    |
| :----- | :----- | :----- | :---- | :---- | :------ |
| Rsvd   | HDTV   | CC     | E     | DACS+ | Encoder |

</div>

<div class="tableblock">

|  Name   | Bit width | Values and Meaning                |
| :-----: | :-------- | :-------------------------------- |
|  SDTV   | 3         | SDTV Format.                      |
|  Rsvd   | 1         | Reserved, set to 0.               |
|  DACs   | 4         | DAC description, lower four bits. |
| Encoder | 8         | Encoder identifier.               |
| TVDACs+ | 4         | DAC description, upper four bits. |
|    E    | 1         | External Communication Port.      |
|   CC    | 2         | Connector Count.                  |
|  HDTV   | 4         | HDTV Format.                      |
|  Rsvd   | 5         | Reserved, set to 0.               |

</div>

<div class="paragraph">

<div class="title">

SDTV Format

</div>

This field determines the default SDTV Format.

</div>

<div class="paragraph">

Currently defined values are:

</div>

<div class="ulist">

  - 0x0 = NTSC\_M (US)

  - 0x1 = NTSC\_J (Japan)

  - 0x2 = PAL\_M (NTSC Timing w/PAL Encoding - Brazilian Format)

  - 0x3 = PAL\_BDGHI (US)

  - 0x4 = PAL\_N (Paraguay and Uruguay Format)

  - 0x5 = PAL\_NC (Argentina Format)

  - 0x6 = Reserved

  - 0x7 = Reserved

</div>

<div class="paragraph">

<div class="title">

DACs

</div>

These bits define the availability of encoder outputs that the board
supports to the TV connectors.

</div>

<div class="tableblock">

| Value | Meaning                                                     |
| :---: | :---------------------------------------------------------- |
| 0x00  | Reserved.                                                   |
| 0x01  | Invalid.                                                    |
| 0x02  | CVBS on Green.                                              |
| 0x03  | CVBS on Green and S-Video on Red (chroma) and Green (luma). |
| 0x04  | CVBS on Blue.                                               |
| 0x05  | Invalid.                                                    |
| 0x06  | Invalid.                                                    |
| 0x07  | CVBS on Blue, S-Video on Red (chroma) and Green (luma).     |
| 0x08  | Standard HDTV.                                              |
| 0x09  | HDTV Twist 1.                                               |
| 0x0A  | SCART.                                                      |
| 0x0B  | Twist 2.                                                    |
| 0x0C  | SCART + HDTV.                                               |
| 0x0D  | Standard HDTV without SDTV.                                 |
| 0x0E  | SCART Twist 1.                                              |
| 0x0F  | SCART + HDTV.                                               |
| 0x11  | Composite + HDTV outputs.                                   |
| 0x12  | HDTV + Scart Twist 1.                                       |
| 0x13  | S-Video on Red (chroma) and Green (luma).                   |

</div>

<div class="paragraph">

<div class="title">

Encoder

</div>

This field describes the exact encoder used on the board.

</div>

<div class="ulist">

  - Brooktree/Conexant
    
    <div class="ulist">
    
      - 0x00 = Brooktree 868
    
      - 0x01 = Brooktree 869
    
      - 0x02 = Conexant 870
    
      - 0x03 = Conexant 871
    
      - 0x04 = Conexant 872
    
      - 0x05 = Conexant 873
    
      - 0x06 = Conexant 874
    
      - 0x07 = Conexant 875
    
    </div>

  - Chrontel
    
    <div class="ulist">
    
      - 0x40 = Chrontel 7003
    
      - 0x41 = Chrontel 7004
    
      - 0x42 = Chrontel 7005
    
      - 0x43 = Chrontel 7006
    
      - 0x44 = Chrontel 7007
    
      - 0x45 = Chrontel 7008
    
      - 0x46 = Chrontel 7009
    
      - 0x47 = Chrontel 7010
    
      - 0x48 = Chrontel 7011
    
      - 0x49 = Chrontel 7012
    
      - 0x4A = Chrontel 7019
    
      - 0x4B = Chrontel 7021
    
    </div>

  - Philips
    
    <div class="ulist">
    
      - 0x80 = Philips 7102
    
      - 0x81 = Philips 7103
    
      - 0x82 = Philips 7104
    
      - 0x83 = Philips 7105
    
      - 0x84 = Philips 7108
    
      - 0x85 = Philips 7108A
    
      - 0x86 = Philips 7108B
    
      - 0x87 = Philips 7109
    
      - 0x88 = Philips 7109A
    
    </div>

  - NVIDIA
    
    <div class="ulist">
    
      - 0xC0 = NVIDIA internal encoder
    
    </div>

</div>

<div class="paragraph">

<div class="title">

TVDACs+

</div>

This field shows bits 4-7 of the TVDACs value.

</div>

<div class="paragraph">

<div class="title">

External Communication Port

</div>

If this device uses external I2C communication, then this field allows
us to know which device will be used. If the device is internal to the
chip, set this bit to 0 as default.

</div>

<div class="paragraph">

Currently defined values are:

</div>

<div class="ulist">

  - 0 = Primary Communications Port

  - 1 = Secondary Communications Port

</div>

<div class="paragraph">

The I2C Control Block Header holds the primary and secondary port
indices.

</div>

<div class="paragraph">

<div class="title">

Connector Count

</div>

Generally, there is only 1 connector per DCB display path. TVs are
special since one output device could have multiple connectors.

</div>

<div class="paragraph">

Currently defined values are:

</div>

<div class="ulist">

  - 0 = 1 Connector

  - 1 = 2 Connectors

  - 2 = 3 Connectors

  - 3 = 4 Connectors

</div>

<div class="paragraph">

If only one bit of either of the Red, Green or Blue defines in the above
DACs field is set, then this field must be set to 1 connector.

</div>

<div class="paragraph">

If two bits of either of the Red, Green or Blue defines in the above
DACs field is set, then this field must be set to 1 or 2 connectors for
a S-Video and/or Composite connector. But those connectors cannot be
displayed simultaneously.

</div>

<div class="paragraph">

If three bits of either of the Red, Green or Blue defines in the above
DACs field is set, then this field must be set to 2 connectors for both
a S-Video and Composite connector.

</div>

<div class="paragraph">

If the HDTV Bit is set, then we can assume that there will be connectors
for YPrPb, S-Video, and Composite off of the [Personal
Cinema](#_personal_cinema_table) pod. So, this field should be set to 3
connectors.

</div>

<div class="paragraph">

<div class="title">

HDTV Format

</div>

This field determines the default HDTV Format.

</div>

<div class="paragraph">

Currently defined values are:

</div>

<div class="ulist">

  - 0x0 = HDTV 576i

  - 0x1 = HDTV 480i

  - 0x2 = HDTV 480p @60Hz

  - 0x3 = HDTV 576p @50Hz

  - 0x4 = HDTV 720p @50Hz

  - 0x5 = HDTV 720p @60Hz

  - 0x6 = HDTV 1080i @50Hz

  - 0x7 = HDTV 1080i @60Hz

  - 0x8 = HDTV 1080p @24Hz

  - 0x9-0xE = Reserved

  - 0xF = Reserved

</div>

<div class="paragraph">

<div class="title">

SDTV Format

</div>

This field determines the default SDTV Format.

</div>

<div class="paragraph">

Currently defined values are:

</div>

<div class="ulist">

  - 0x0 = NTSC\_M (US)

  - 0x1 = NTSC\_J (Japan)

  - 0x2 = PAL\_M (NTSC Timing w/PAL Encoding - Brazilian Format)

  - 0x3 = PAL\_BDGHI (US)

  - 0x4 = PAL\_N (Paraguay and Uruguay Format)

  - 0x5 = PAL\_NC (Argentina Format)

  - 0x6 = Reserved

  - 0x7 = Reserved

</div>

<div class="paragraph">

<div class="title">

Extra Information

</div>

The TV sub-DACs are labeled 0, 1, 2, and 3 for G80 and later GPUs. There
is no plan currently to support external TV encoders. When the CRT is
used, sub-DAC 0 is used for Red, sub-DAC 1 is used for Green and sub-DAC
2 is used for Blue, always.

</div>

<div class="paragraph">

This table should explain how each TVDACs value corresponds to each
sub-DAC TV Protocol:

</div>

<div class="tableblock">

TVDACs

</div>

</div>

Composite

S-Video

HDTV

SCART

0

1

2

3

0

1

2

3

0

1

2

3

0

1

2

3

0

Invalid

1

Invalid

<span class="red">2</span>

CVBS

<span class="red">3</span>

CVBS

C

Y

4

CVBS

5

Invalid

6

Invalid

<span class="red">7</span>

CVBS

C

Y

<span class="red">8</span>

CVBS

C

Y

R/Pr

G/Y

B/Pb

9

CVBS

Y

C

R/Pr

G/Y

B/Pb

A

CVBS

Y

C

G

B

R

CVBS

B

CVBS

C

Y

R/Pr

G/Y

B/Pb

C

CVBS

Y

C

G/Y

B/Pb

R/Pr

G

B

R

CVBS

<span class="red">D</span>

R/Pr

G/Y

B/Pb

E

CVBS

Y

C

R

G

B

CVBS

F

CVBS

Y

C

G/Y

R/Pr

B/Pb

R

G

B

CVBS

0x10

Reserved

0x11

CVBS

R/Pr

G/Y

B/Pb

<span class="red">0x12</span>

CVBS<sup>1</sup>

CVBS<sup>2</sup>

Y

C

G/Y

B/Pb

R/Pr

CVBS

G

R

B

<span class="red">0x13</span>

Y

C

<div class="paragraph">

<span class="red">Only the entries above in red are currently supported
in Core6+.</span>

</div>

<div class="paragraph">

Here’s how we choose the connector types based on the load:

</div>

<div class="ulist">

  - The SCART configuration is chosen if SCART is valid for the board’s
    TVDACs setting, and loads are detected on all four sub-DACS.

  - The HDTV configuration is chosen if HDTV is valid for the board’s
    TVDACs setting, SCART was not chosen, and loads are detected on the
    three sub-DACS that are specified to carry signals for the HDTV
    configuration.

  - The S-Video configuration is chosen if S-Video is valid for the
    board’s TVDACs setting, SCART or HTDV were not chosen, and loads are
    detected on the two sub-DACS that are specified to carry signals for
    the S-Video configuration.

  - The Composite configuration is chosen if Composite is valid for the
    board’s TVDACs setting while SCART, HTDV, or S-Video were not
    chosen.

  - Some configurations allow for two different Composite/CVBS signals.
    CVBS1 is used if that DAC has a load. Otherwise, we use CVBS2. CVBS1
    is the CVBS signal when the 4-pin S-Video to CVBS dongle is used.
    CVBS2 is the CVBS signal when the 7-pin HDTV component dongle is
    used (the B/Pb connector on the HDTV component RCA connectors on the
    7-pin dongle is labeled as "Comp" for use with CVBS).

</div>

<div class="ulist">

<div class="title">

Additional Notes

</div>

  - The S-Video Y signal will always follow the G/Y signal on the 7-pin
    HDTV component dongle (because the pins match up on the connectors).

  - The S-Video C signal will always follow the R/Pr signal on the 7-pin
    HDTV component dongle (because the pins match up on the connectors).

  - The CVBS (Composite) signal will always follow the B/Pb signal on
    the 7-pin HDTV component dongle (because the B/Pb connector is
    labeled for use as Composite (CVBS)).

  - The CVBS (Composite) signal will always follow the Y signal on the
    4-pin S-Video to CVBS dongle (because the dongle has the RCA CVBS
    signal wired that way).

</div>

<div class="sect1">

## Communications Control Block

<div class="sectionbody">

<div class="paragraph">

This structure is REQUIRED in the DCB 4.x spec. It must be listed inside
every DCB. The VBIOS and the (U)EFI driver will use the data from this
structure.

</div>

<div class="paragraph">

The Communications Control Block provides logical to physical
translation of all the different ways that the GPU can use to
communicate with other devices on the board or to displays. Prior to DCB
4.0 there were 3 different I2C Ports for GPUs and an extra 2 for Crush
(nForce chipset) 11/17. The Northbridge, which holds the integrated GPU,
only has 1.5 V signaling, but the DDC/EDID spec requires 3.3 V
signaling. So, for Crush, we use two ports on the south bridge to handle
the DDC voltage requirements.

</div>

<div class="admonitionblock">

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td><div class="title">
Note
</div></td>
<td>Crush, also known as nForce or nForce2, is a motherboard chipset created by NVIDIA. Crush was released in mid-2001.</td>
</tr>
</tbody>
</table>

</div>

<div class="paragraph">

For DCB 4.x, the norm will be 4 I2C ports as exposed on G80. With
Display Port added in G98, we’ll expose a DPAUX port as well.

</div>

<div class="sect2">

### Communications Control Block 0x40

<div class="paragraph">

Version 0x40 of the Communications Control Block, which is used for Core
6, and Core 6 revision 2, Core70, Core80, and Core82 (which associate to
G8x, G9x, GT2xx, GF1xx, GKxxx, and GM10x GPUs) is described below.

</div>

<div class="sect3">

#### Communications Control Block 0x40 Header

<div class="tableblock">

|             Name             | Bit width | Values and Meaning                                                                                                                                                                             |
| :--------------------------: | :-------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|           Version            | 8         | Version \# of the CCB Header and Entries. CCB 4.0 will start with a value of 0x40 here. A version of 0 here is invalid.                                                                        |
|         Header Size          | 8         | Size of the CCB Header in bytes. This is typically 5 bytes.                                                                                                                                    |
|         Entry Count          | 8         | Number of CCB Entries starting directly after the end of this table.                                                                                                                           |
|          Entry Size          | 8         | Size of each entry in bytes. This field should be 4.                                                                                                                                           |
|  Primary Communication Port  | 4         | Index for the primary communications port. Specifically, if we need to talk with an external device, the port referenced by this index will be the primary port to talk with that device.      |
| Secondary Communication Port | 4         | Index for the secondary communications port. Specifically, if we need to talk with an external device, this port referenced by this index will be the secondary port to talk with that device. |

</div>

<div class="paragraph">

There is one port entry for each port used. A DVI-I connector’s two
device entries share the same I2C port.

</div>

</div>

<div class="sect3">

#### Communications Control Block 0x40 Entry

<div class="paragraph">

<div class="title">

Access Method

</div>

The first upper 8 bits of each entry is called the Access Method. This
field indicates how the software should control each port. From NV50
onward a new port mapping was implemented. Older I2C Access methods -
CRTC indexed mapping and PCI IO Mapping - have been removed, but their
values reserved to allow SW compatibility. Here’s the NV50 and later
Defined Access Methods:

</div>

<div class="tableblock">

| Value | Method                                 |
| :---- | :------------------------------------- |
| 0     | Reserved (Prior DCB Usage)             |
| 1     | Reserved (Prior DCB Usage)             |
| 2     | Reserved (Prior DCB Usage)             |
| 3     | Reserved (Prior DCB Usage)             |
| 4     | Reserved (Prior DCB Usage)             |
| 5     | I2C Access Method                      |
| 6     | Display Port AUX Channel Access Method |

</div>

<div class="sect4">

##### I2C Access Method

<div class="tableblock">

31

</div>

</div>

</div>

</div>

</div>

</div>

30

29

28

27

26

25

24

23

22

21

20

19

18

17

16

15

14

13

12

11

10

9

8

7

6

5

4

3

2

1

0

I2CAccess = 5

Reserved

Rsv

DP

H

Speed

Phys Port

<div class="tableblock">

<table style="width:99%;">
<colgroup>
<col style="width: 21%" />
<col style="width: 7%" />
<col style="width: 71%" />
</colgroup>
<thead>
<tr class="header">
<th style="text-align: center;">Name</th>
<th style="text-align: left;">Bit width</th>
<th style="text-align: left;">Values and Meaning</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: center;"><p>Physical Port</p></td>
<td style="text-align: left;"><p>4</p></td>
<td style="text-align: left;"><p>Physical Nv5x Port</p>
<p>* 0 = DDC0 * 1 = DDC1 * 2 = DDC2 * 3 = I2C</p></td>
</tr>
<tr class="even">
<td style="text-align: center;"><p>Port Speed</p></td>
<td style="text-align: left;"><p>4</p></td>
<td style="text-align: left;"><p>The I2C spec defines 3 different communication speeds: * Standard - 100 kHz * Fast - 400 kHz * High Speed - 3.4 MHz</p>
<p>Each device on an I2C bus must comply with that speed otherwise, the lowest device on that bus will clock stall the speed to what it can handle. High Speed requires extra programming to allow a specific master to send the high speed data. There are programming requirements to also allow for the fallback between higher level speeds and lower levels speeds.</p>
<p>No traffic on the I2C port may exceed the speed specified here.</p>
<p>Most (perhaps all) DCBs set this field to 0. The currently defined levels are:</p>
<p>* 0x0 = Use Defaults (Probably the only one we’ll ever use.) * 0x1 = 100 kHz as per Standard specification * 0x2 = 200 kHz * 0x3 = 400 kHz as per Fast specification * 0x4 = 800 kHz * 0x5 = 1.6 MHz * 0x6 = 3.4 MHz as per High Speed specification * 0x7 = 60 KHz</p></td>
</tr>
<tr class="odd">
<td style="text-align: center;"><p>Hybrid Pad</p></td>
<td style="text-align: left;"><p>1</p></td>
<td style="text-align: left;"><p>This bit is used to tell us if we’re enabling Hybrid Pad control for this entry. Hybrid pad control requires that we switch bits in the NV_PMGR_HYBRID_PADCTL area when switching between I2C output and DPAux output. The values here are:</p>
<p>* 0 = Normal Mode - Generic I2C Port * 1 = Hybrid Mode - Pad allows for switching between DPAux and I2C</p></td>
</tr>
<tr class="even">
<td style="text-align: center;"><p>Physical DP Aux Port</p></td>
<td style="text-align: left;"><p>4</p></td>
<td style="text-align: left;"><p>This is the physical DP Aux port used only when Hybrid Pad field is in Hybrid Mode. We need this value since NV_PMGR_HYBRID_PADCTL is indexed based on the DP Port value.</p></td>
</tr>
<tr class="odd">
<td style="text-align: center;"><p>Reserved</p></td>
<td style="text-align: left;"><p>11</p></td>
<td style="text-align: left;"><p>Set as 0.</p></td>
</tr>
<tr class="even">
<td style="text-align: center;"><p>I2C Access Method</p></td>
<td style="text-align: left;"><p>8</p></td>
<td style="text-align: left;"><p>Must be set to 5 for this Access Method</p></td>
</tr>
</tbody>
</table>

</div>

<div class="sect4">

##### Display Port AUX Channel Access Method

<div class="tableblock">

31

</div>

</div>

30

29

28

27

26

25

24

23

22

21

20

19

18

17

16

15

14

13

12

11

10

9

8

7

6

5

4

3

2

1

0

DP Aux Access = 6

Reserved

Rsvd

I2C

H

Rsvd

Phys Port

<div class="tableblock">

| Name | Bit width | Values and Meaning                         |
| :--: | :-------- | :----------------------------------------- |
| Port | 4         | Physical Display Ports                     |
|  H   | 1         | Hybrid Pad                                 |
| I2C  | 4         | Physical I2C Port                          |
| Rsvd | 11        | Reserved. Set as 0.                        |
| Type | 8         | Display Port AUX Channel Access Method = 6 |

</div>

<div class="ulist">

<div class="title">

Physical Display Port mappings:

</div>

  - 0 = AUXCH 0

  - 1 = AUXCH 1

  - 2 = AUXCH 2

  - 3 = AUXCH 3

</div>

<div class="paragraph">

<div class="title">

Hybrid Pad

</div>

This bit is used to tell us if we’re enabling Hybrid Pad control for
this entry. Hybrid pad control requires that we switch bits in the
NV\_PMGR\_HYBRID\_PADCTL area when switching between I2C output and
DPAux output. The values here are:

</div>

<div class="ulist">

  - 0 = Normal Mode - Generic I2C Port

  - 1 = Hybrid Mode - Pad allows for switching between DPAux and I2C

</div>

<div class="paragraph">

<div class="title">

Physical I2C Port

</div>

This is the physical I2C port used only when Hybrid Pad field is in
Hybrid Mode.

</div>

<div class="paragraph">

<div class="title">

Type

</div>

Must be set to 6 to indicate Display Port AUX Channel Access Method

</div>

<div class="sect2">

### Communications Control Block 0x41

<div class="paragraph">

Version 0x41 of the Communications Control Block, which will be used for
GM20x+ or Core 84 and future cores, is listed below.

</div>

<div class="sect3">

#### Communications Control Block 0x41 Header

<div class="tableblock">

|             Name             | Bit width | Values and Meaning                                                                                                                                                                             |
| :--------------------------: | :-------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|           Version            | 8         | Version \# of the CCB Header and Entries. CCB 4.1 will start with a value of 0x41 here. A version of 0 here is invalid.                                                                        |
|         Header Size          | 8         | Size of the CCB Header in bytes. In CCB 4.1 this is 6 bytes.                                                                                                                                   |
|         Entry Count          | 8         | Number of CCB Entries starting directly after the end of this table.                                                                                                                           |
|          Entry Size          | 8         | Size of each entry in bytes. This field should be 4.                                                                                                                                           |
|  Primary Communication Port  | 8         | Index for the primary communications port. Specifically, if we need to talk with an external device, the port referenced by this index will be the primary port to talk with that device.      |
| Secondary Communication Port | 8         | Index for the secondary communications port. Specifically, if we need to talk with an external device, this port referenced by this index will be the secondary port to talk with that device. |

</div>

<div class="paragraph">

There is one port entry for each port used. A DVI-I connector’s two
device entries share the same I2C port.

</div>

</div>

<div class="sect3">

#### Communications Control Block 0x41 Entry

<div class="paragraph">

There is one CCB entry for each set of communications lines (a "pad") on
the board. For example, A DVI-I connector’s two device entries share the
same I2C port so they point to the same CCB entry. DP and TMDS entries
that are "partnered" (share the same connector and pads) also share the
same CCB entry.

</div>

<div class="tableblock">

<table style="width:99%;">
<colgroup>
<col style="width: 21%" />
<col style="width: 7%" />
<col style="width: 71%" />
</colgroup>
<thead>
<tr class="header">
<th style="text-align: center;">Name</th>
<th style="text-align: left;">Bit width</th>
<th style="text-align: left;">Values and Meaning</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: center;"><p>I2C Port</p></td>
<td style="text-align: left;"><p>5</p></td>
<td style="text-align: left;"><p>Index in PMGR for the I2C Controller that drives the physical pad denoted by this CCB entry. The value 0x1F denotes Unused, meaning that this pad does not support I2C.</p></td>
</tr>
<tr class="even">
<td style="text-align: center;"><p>DPAUX Port</p></td>
<td style="text-align: left;"><p>5</p></td>
<td style="text-align: left;"><p>Index in PMGR for the DPAUX Controller that drives the physical pad denoted by this CCB entry. The value 0x1F denotes Unused, meaning that this pad does not support DPAUX.</p></td>
</tr>
<tr class="odd">
<td style="text-align: center;"><p>Reserved</p></td>
<td style="text-align: left;"><p>18</p></td>
<td style="text-align: left;"><p>Set as 0.</p></td>
</tr>
<tr class="even">
<td style="text-align: center;"><p>I2C Port Speed</p></td>
<td style="text-align: left;"><p>4</p></td>
<td style="text-align: left;"><p>The I2C spec defines 3 different communication speeds: * Standard - 100 kHz * Fast - 400 kHz * High Speed - 3.4 MHz</p>
<p>Each device on an I2C bus must comply with that speed otherwise, the lowest device on that bus will clock stall the speed to what it can handle. High Speed requires extra programming to allow a specific master to send the high speed data. There are programming requirements to also allow for the fallback between higher level speeds and lower levels speeds.</p>
<p>No traffic on the I2C port may exceed the speed specified here.</p>
<p>Most (perhaps all) DCBs set this field to 0. The currently defined levels are:</p>
<p>* 0x0 = Use Defaults (Probably the only one we’ll ever use.) * 0x1 = 100 kHz as per Standard specification * 0x2 = 200 kHz * 0x3 = 400 kHz as per Fast specification * 0x4 = 800 kHz * 0x5 = 1.6 MHz * 0x6 = 3.4 MHz as per High Speed specification * 0x7 = 60 KHz * 0x8 = 300 kHz</p></td>
</tr>
</tbody>
</table>

</div>

</div>

</div>

<div class="sect1">

## Input Devices Table

<div class="sectionbody">

<div class="paragraph">

This structure is optional. It only needs to be defined if the board
provides input devices. Also, the VBIOS or FCODE does not need to use
this structure. Only the drivers will use it.

</div>

<div class="paragraph">

The Input Devices are listed at a location in the ROM dictated by the
16-bit Input Devices Pointer listed in the DCB Header. Currently, the
maximum number of devices is 8. Each device is listed in one 8-bit
entry.

</div>

<div class="paragraph">

If a device has an Input Device Structure, but not a [Personal
Cinema](#_personal_cinema_table) Structure defined, we treat that board
as a generic VIVO (Video-In, Video-Out) board.

</div>

<div class="paragraph">

It is assumed that each of these Input Devices is controlled via I2C
through the Primary Communications Port.

</div>

<div class="sect2">

### Input Devices Header

<div class="tableblock">

|    Name     | Bit width | Values and Meaning                                                                                  |
| :---------: | :-------- | :-------------------------------------------------------------------------------------------------- |
|   Version   | 8         | Version \# of the Input Devices Header and Entries. Input Devices 4.0 start with a version of 0x40. |
| Header Size | 8         | Size of the Input Devices Header in Bytes. Initially, this is 4 bytes.                              |
| Entry Count | 8         | Number of Input Devices Entries starting directly after the end of this table.                      |
| Entry Size  | 8         | Size of Each Entry in bytes. This field should be 1.                                                |

</div>

</div>

<div class="sect2">

### Input Device Entry

<div class="tableblock">

7

</div>

</div>

</div>

</div>

6

5

4

3

2

1

0

VT

T

Mode

<div class="tableblock">

<table style="width:99%;">
<colgroup>
<col style="width: 12%" />
<col style="width: 4%" />
<col style="width: 83%" />
</colgroup>
<thead>
<tr class="header">
<th style="text-align: center;">Name</th>
<th style="text-align: left;">Bit width</th>
<th style="text-align: left;">Values and Meaning</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: center;"><p>Mode</p></td>
<td style="text-align: left;"><p>4</p></td>
<td style="text-align: left;"><p>This field lists the Mode number that this device supports. If we encounter a Mode of 0xF, that signifies a Skip Entry. This allows for quick removal of a specific entry from the Input Devices.</p></td>
</tr>
<tr class="even">
<td style="text-align: center;"><p>Type</p></td>
<td style="text-align: left;"><p>2</p></td>
<td style="text-align: left;"><p>This field describes the type of input device that is connected. Current defined possible values are:</p>
<p>* 0 = VCR, * 1 = TV</p></td>
</tr>
<tr class="odd">
<td style="text-align: center;"><p>Video Type</p></td>
<td style="text-align: left;"><p>2</p></td>
<td style="text-align: left;"><p>This field describes the video type of input device that is connected.</p>
<p>Currently defined values are:</p>
<p>* 0 = CVBS, * 1 = Tuner, * 2 = S-Video</p></td>
</tr>
</tbody>
</table>

</div>

<div class="sect1">

## Personal Cinema Table

<div class="sectionbody">

<div class="admonitionblock">

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td><div class="title">
Note
</div></td>
<td>"Personal Cinema" refers to a line of graphics boards with pre-G80 NVIDIA GPUs and on-board television tuners.</td>
</tr>
</tbody>
</table>

</div>

<div class="paragraph">

This structure is optional. It only needs to be defined if the board is
intending to provide Personal Cinema support. The VBIOS or FCODE does
not need to use this structure. Only the drivers will use it.

</div>

<div class="paragraph">

There are many specific defines needed for the personal cinema in order
to know which devices are available. Because there are no entries needed
for this table, the normal Entry Count and Entry Size will not be a part
of this table for now.

</div>

<div class="paragraph">

If both the Board ID and the Vendor ID are 0, then the Personal Cinema
Table data should be considered invalid. This is akin to other table’s
SKIP ENTRY, meaning that we should just skip this table if these IDs are
both 0.

</div>

<div class="paragraph">

If a device has an [Input Devices Table](#_input_devices_table), but not
a Personal Cinema Structure defined, we treat that board as a generic
VIVO (Video-In, Video-Out) board.

</div>

<div class="paragraph">

It is assumed that each of these Personal Cinema Devices is controlled
via I2C through the Primary Communications Port.

</div>

<div class="sect2">

### Personal Cinema Table Structure

<div class="tableblock">

31

</div>

</div>

</div>

</div>

30

29

28

27

26

25

24

23

22

21

20

19

18

17

16

15

14

13

12

11

10

9

8

7

6

5

4

3

2

1

0

Vendor ID

Board ID

Header Size

Version

<div class="tableblock">

63

</div>

62

61

60

59

58

57

56

55

54

53

52

51

50

49

48

47

46

45

44

43

42

41

40

39

38

37

36

35

34

33

32

IRCtrl

PwrCtrlIC

Demod1

ATuner 1

SndDcd1

Std

Eio

<div class="tableblock">

95

</div>

94

93

92

91

90

89

88

87

86

85

84

83

82

81

80

79

78

77

76

75

74

73

72

71

70

69

68

67

66

65

64

Demod2

R

T2F

R

T1F

ATuner2

Rsvd

SndDcd2

<div class="tableblock">

|    Name     | Bit width | Values and Meaning                                      |
| :---------: | :-------- | :------------------------------------------------------ |
|   Version   | 8         | Version = 0x40                                          |
| Header Size | 8         | Size in bytes, 12 for v4.0                              |
|  Board ID   | 8         | Personal Cinema Board ID for this board                 |
|  Vendor ID  | 8         | Vendor ID for this board                                |
|     Eio     | 2         | Expander IO bus width                                   |
|    TVStd    | 2         | TV Standard used e.g. NTSC or PAL                       |
|   SndDec1   | 4         | Sound Decoder \#1 ID                                    |
|   ATuner1   | 8         | Analog Tuner \#1 type, the first analog tuner           |
|   Demod1    | 8         | Demodulator \#1, the first digital-signal tuner         |
|  PwrCtrlIC  | 4         | Satellite Dish power controller IC type                 |
|   IRCtrl    | 4         | The InfraRed transmitter microcontroller type           |
|   SndDec2   | 4         | Sound Decoder \#2 ID                                    |
|    Rsvd     | 4         | Reserved, set to 0                                      |
|   ATuner2   | 8         | Analog Tuner \#2 type.                                  |
|     T1F     | 3         | Tuner \#1 Functionality, digitial TV, analog TV and FM. |
|  Reserved   | 1         | Reserved, set to 0                                      |
|     T2F     | 3         | Tuner \#2 Functionality                                 |
|  Reserved   | 1         | Reserved, set to 0                                      |
|   Demod2    | 8         | Demodulator \#2, the second digital-signal tuner        |

</div>

<div class="paragraph">

<div class="title">

Version

</div>

Version \# of the Personal Cinema Header. The original Personal Cinema
table version will start with a value of 0x40 here. If the version is 0
here, then the driver will assume that this table is invalid.

</div>

<div class="paragraph">

<div class="title">

Header Size

</div>

Size of the Personal Cinema Header in bytes. This is 12 bytes for v4.0.

</div>

<div class="paragraph">

<div class="title">

Board ID

</div>

This field lists the Personal Cinema Board ID for this board. This
provides a mechanism for SW to differentiate between individual Personal
Cinema boards and generic Video-In-Video-Out (VIVO) boards.

</div>

<div class="paragraph">

Currently defined values are:

</div>

<div class="ulist">

  - 0x00 = Generic VIVO board or No Personal Cinema Support

  - 0x01 = P79

  - 0x02 = P104

  - 0x03 = P164-NV31

  - 0x04 = P164-NV34

  - 0x05 = P186-NV35

  - 0x06 = P187-NV35

  - 0x07 = P178-NV36

  - 0x08 = P253-NV43

  - 0x09 = P254-NV44

  - 0x0A = P178-NV36-A2M

  - 0x0B = P293

  - 0x0C = P178-NV36-FPGA

  - 0x0D = P143-NV34-FPGA

  - 0x0E = P143-NV34-Non-FPGA

  - 0x10 = P256-NV43

  - 0x11 = Compro

  - 0x13 = P274-NV41

  - 0x21 = Asus AIO

  - 0x22 = Asus external tuner

  - 0x30 = Customer Reserved 0

  - 0x31 = Customer Reserved 1

  - 0x32 = Customer Reserved 2

</div>

<div class="paragraph">

<div class="title">

Vendor ID

</div>

This field lists the Personal Cinema Vendor ID for this board. Current
defined possible values are:

</div>

<div class="ulist">

  - 0x00 = Generic VIVO board or No Personal Cinema Support

  - 0xde = NVIDIA

  - 0xcb = Compro

  - 0x81 = Asus

</div>

<div class="paragraph">

<div class="title">

Expander IO

</div>

This field describes the exact number of bits used for the expander IO
bus.

</div>

<div class="paragraph">

Currently defined values are:

</div>

<div class="ulist">

  - 0 = None or Not Applicable

  - 1 = 8 bits

  - 2 = 16 bits

  - 3 = RF remote

</div>

<div class="paragraph">

<div class="title">

TV Standard

</div>

This field describes the TV standard used for the input devices.

</div>

<div class="paragraph">

Currently defined values are:

</div>

<div class="ulist">

  - 0 = NTSC

  - 1 = PAL/SECAM

  - 2 = Worldwide

  - 3 = Reserved

</div>

<div class="paragraph">

<div class="title">

Sound Decoder \#1

</div>

This field describes the first Sound Decoder used on the board. Current
defined possible values are:

</div>

<div class="ulist">

  - 0 = Mono

  - 2 = A2 (TDA9873)

  - 3 = NICAM (TDA9874)

  - 4 = BTSC (TDA9850)

  - 5 = FM-FM Japan (TA8874z)

  - 6 = BTSC/EIAJ (SAA7133/SAA7173)

  - 7 = A2,NICAM (SAA7134/SAA7174)

  - 8 = Worldwide (SAA7135/SAA7175)

  - 9 = Micronas MSP 3425G (NTSC)

  - 10 = Micronas MSP 3415G (PAL)

  - 11 = SAA7174A

  - 12 = SAA7171

  - 15 = Not Present

</div>

<div class="paragraph">

<div class="title">

Tuner Type \#1

</div>

This field describes the first analog-signal tuner used on the board.

</div>

<div class="paragraph">

Currently defined values are:

</div>

<div class="ulist">

  - 0x00 = Not Present

  - 0x01 = Philips FI1216 MK2

  - 0x02 = Philips FI1216 MF

  - 0x03 = Philips FI1236 MK2

  - 0x04 = Philips FI1246 MK2

  - 0x05 = Philips FI1256 MK2

  - 0x06 = Philips FQ1216 ME

  - 0x07 = Philips FQ1216 ME MK3

  - 0x08 = Philips FQ1236 ME MK3

  - 0x09 = Philips TDA 8275

  - 0x11 = Temic 4036FY5,4032FY5

  - 0x12 = Temic 4006FH5,4002FH5

  - 0x13 = Temic 4066FY5,4036FY5

  - 0x14 = Temic 4016FY5,4012FY5

  - 0x15 = Temic 4136

  - 0x16 = Temic 4146

  - 0x17 = Microtune MT2040

  - 0x18 = Microtune MT2050

  - 0x19 = Microtune 7102DT5

  - 0x20 = Microtune 7132DT5

  - 0x21 = Microtune MT2060

  - 0x22 = Microtune 4039FR5

  - 0x23 = Microtune 4049FM5

  - 0x30 = LG TALN-M200T (PAL)

  - 0x31 = LG TALN-H200T (NTSC)

  - 0x32 = TALN-S200T (SECAM L/L' & PAL B/G, I/I, D/K)

  - 0x60 = Samsung TEBN9282PK01A

  - 0x81 = Philips FM1216

  - 0x82 = Philips FM1216MF

  - 0x83 = Philips FM1236

  - 0x84 = Philips FM1246

  - 0x85 = Philips FM1256

  - 0x86 = Philips FM1216 ME

  - 0x87 = Philips FM1216 ME MK3

  - 0x88 = Philips FM1236 ME MK3

</div>

<div class="paragraph">

<div class="title">

Demodulator \#1

</div>

The first digital-signal tuner used this board. This field has these hex
defines:

</div>

<div class="ulist">

  - 0x00 = Not present

  - 0x01 = TDA9885 (PAL/NTSC Analog)

  - 0x02 = TDA9886 (PAL/NTSC/SECAM Analog)

  - 0x03 = TDA9887 (PAL/NTSC/SECAM QSS Analog)

  - 0x04 = Philips SAA7171

  - 0x10 = Conexant CX24121

  - 0x15 = Phillips TDA8260TW

  - 0x16 = Zarlink MT352

  - 0x17 = LGDT3302

  - 0x18 = Micronas DRX3960A

</div>

<div class="paragraph">

<div class="title">

Power Control IC

</div>

Satellite Dish power controller. This field has these hex defines:

</div>

<div class="ulist">

  - 0 = Not present

  - 1 = LNBP21 - I2C Address 0x10

</div>

<div class="paragraph">

<div class="title">

Microcontroller

</div>

The microcontroller chip used for infrared (IR) transmitting to control
other IR devices. This field has these values

</div>

<div class="ulist">

  - 0 = Not present

  - 6 = PIC12F629

  - 7 = PIC12CE673

</div>

<div class="paragraph">

<div class="title">

Sound Decoder \#2

</div>

This field describes a possible second Sound Decoder used on the board.
The values are the same as with Sound Decoder \#1.

</div>

<div class="paragraph">

<div class="title">

Tuner Type \#2

</div>

This field describes a possible second analog-signal tuner used on the
board. The values are the same as with Tuner Type \#1.

</div>

<div class="paragraph">

<div class="title">

Tuner \#1 Functionality

</div>

This field describes the functionality supported by Tuner \#1.

</div>

<div class="paragraph">

Currently defined values are:

</div>

<div class="ulist">

  - 0 = None

  - 1 = Digital TV

  - 2 = Analog TV

  - 3 = Analog + Digital TV

  - 4 = FM

  - 5 = Digital + FM

  - 6 = Analog + FM

  - 7 = Analog + Digital + FM

</div>

<div class="paragraph">

<div class="title">

Tuner \#2 Functionality

</div>

This field describes the functionality allowed by Tuner Type \#2 field.
The currently defined values are the same as those for Tuner \#1
Functionality.

</div>

<div class="paragraph">

<div class="title">

Demodulator \#2

</div>

The possible second digital-signal tuner used this board. This field has
the same defines as Demodulator \#1.

</div>

<div class="sect1">

## GPIO Assignment Table

<div class="sectionbody">

<div class="paragraph">

The GPIO Assignment table creates a logical mapping of function-based
usage names to physical GPIOs within the GPU. Each pin has

</div>

<div class="ulist">

  - a logical ON State and

  - a logical OFF State.

</div>

<div class="paragraph">

Each state can be distinctly defined physically via:

</div>

<div class="ulist">

  - Sending output high to the GPIO,

  - Sending output low to the GPIO, or

  - Tristating the GPIO (Setting it to Input Mode).

</div>

<div class="paragraph">

Alternately, specific GPIOs can also be assigned to carry Pulse Width
Modulated (PWM) signals. This can be used for fan speed control or
backlight power control.

</div>

<div class="paragraph">

This table is required in all ROMs. It must be listed inside every DCB.
The VBIOS and the FCODE will use the data from this structure.

</div>

<div class="sect2">

### GPIO Assignment Table Header

<div class="paragraph">

When moving to GF110, the HW team merged the Normal/Alternate/Sequencer
modes of the GPIO into one 8 bit field in a GPIO register. In order to
better manage that change, we decided to increase the revision from the
initial 0x40 version to 0x41 and re-organize the bit fields in each GPIO
table entry to accommodate a new field that matches the field in the HW
register directly.

</div>

<div class="paragraph">

Version 0x41, as used for GF11x+ / Core75 and future cores, is listed
below.

</div>

<div class="tableblock">

Table 3. GPIO Assignment Table Header Version 4.1

31

</div>

</div>

</div>

</div>

30

29

28

27

26

25

24

23

22

21

20

19

18

17

16

15

14

13

12

11

10

9

8

7

6

5

4

3

2

1

0

Entry Size = 5

Entry Count

Header Size = 6

Version = 0x41

<div class="tableblock">

47

</div>

46

45

44

43

42

41

40

39

38

37

36

35

34

33

32

GPIOAssTabPtr

<div class="paragraph">

<div class="title">

Version

</div>

Version \# of the GPIO Assignment Table Header and Entries. The current
GPIO Assignment Table version is 4.1 or a value of 0x41 in this field.
If this version is 0, then the driver will assume that this table is
invalid.

</div>

<div class="paragraph">

<div class="title">

Header Size

</div>

Size of the GPIO Assignment Table in bytes. For version 4.1 this is 6
bytes.

</div>

<div class="paragraph">

<div class="title">

Entry Count

</div>

Number of GPIO Assignment Table Entries starting directly after the end
of this header.

</div>

<div class="paragraph">

<div class="title">

Entry Size

</div>

Size of Each Entry in bytes. For version 4.0, this was 4 bytes. For
version 4.1, this is now 5 bytes.

</div>

<div class="paragraph">

<div class="title">

External GPIO Assignment Table Master Header Pointer

</div>

Pointer to the [External GPIO Assignment Master
Table](#_external_gpio_assignment_master_table). This field can be set
to 0 to indicate no support for this table.

</div>

<div class="sect2">

### GPIO Assignment Table Entry

<div class="paragraph">

Please note that this structure below is for version 4.1.

</div>

<div class="tableblock">

31

</div>

</div>

30

29

28

27

26

25

24

23

22

21

20

19

18

17

16

15

14

13

12

11

10

9

8

7

6

5

4

3

2

1

0

PM

R

GS

Input HW Select

Output HW Select

Function

I

IO

PinNum

<div class="tableblock">

39

</div>

38

37

36

35

34

33

32

OE

OT

FE

FT

LockPin

<div class="tableblock">

|       Name       | Bit width | Values and Meaning               |
| :--------------: | :-------- | :------------------------------- |
|      PinNum      | 6 (5:0)   | GPIO Pin Number                  |
|        IO        | 1 (6:6)   | I/O Type                         |
|        I         | 1 (7:7)   | Initialize pin state             |
|     Function     | 8 (15:8)  |                                  |
| Output HW select | 8 (23:16) | Output hardware function setting |
| Input HW select  | 5 (28:24) | Input hardware function setting  |
|        GS        | 1 (29:29) | GSYNC Header                     |
|        R         | 1 (30:30) | Reserved                         |
|        PM        | 1 (31:31) | Pulse Width Modulate             |
|     LockPin      | 4 (35:32) | Lock Pin Number                  |
|        FT        | 1 (36:36) | Off Data                         |
|        FE        | 1 (37:37) | Off Enable                       |
|        OT        | 1 (38:38) | On Data                          |
|        OE        | 1 (39:39) | On Enable                        |

</div>

<div class="paragraph">

<div class="title">

GPIO Number (5:0)

</div>

The GPIO number associated with this entry. Older chips have a maximum
of 9 GPIO pins. G80+ have 15 GPIOs in register space. This field must be
0 if the I/O Type field is set to
NV\_GPIO\_IO\_TYPE\_DEDICATED\_LOCK\_PIN.

</div>

<div class="paragraph">

<div class="title">

I/O Type

</div>

The I/O Type field is used to specify if this entry represents an actual
GPIO or instead represents a similar type of entity. This field is an
enumeration that currently has the following values:

</div>

<div class="ulist">

  - 0 = NV\_GPIO\_IO\_TYPE\_GPIO - This entry represents a normal
    internal GPIO.

  - 1 = NV\_GPIO\_IO\_TYPE\_DEDICATED\_LOCK\_PIN - This entry represents
    an internal dedicated lock pin. No actual GPIO is associated with
    the lock pin. The GPIO Number field must be set to zero.

</div>

<div class="paragraph">

<div class="title">

Initialize State (Init) (7:7)

</div>

This field specifies the initial state to set the GPIO to during boot.
If this bit is 0, then the software will initialize the GPIO at boot to
the settings specified by "Off Data" and "Off Enable". If this bit is 1,
then the software will initialize the GPIO at boot to the settings
specified by "On Data" and "On Enable".

</div>

<div class="paragraph">

<div class="title">

Function (15:8)

</div>

This lists the function of each GPIO pin. Here’s a list of the function
numbers and a short description of each:

</div>

<div class="ulist">

  - 0 = LCD0 backlight: Backlight control. LCD0 corresponds to the LCD0
    defined in the LCD ID field in the Connector Table.

  - 1 = LCD0 power: Panel Power control. LCD0 corresponds to the LCD0
    defined in the LCD ID field in the Connector Table.

  - 2 = LCD0 Power Status: Panel Power status. LCD0 corresponds to the
    LCD0 defined in the LCD ID field in the Connector Table.

  - 3 = VSYNC: Alternate VSync signal using GPIO pin.

  - 4 = VSEL0: Voltage Select Bit 0

  - 5 = VSEL1: Voltage Select Bit 1

  - 6 = VSEL2: Voltage Select Bit 2

  - 7 = Hotplug A: 1st Hotplug signal

  - 8 = Hotplug B: 2nd Hotplug signal

  - 9 = Fan: Fan control. Can be on or off, or pulse width modulation to
    control speed.

  - 10 = Reserved

  - 11 = Reserved

  - 12 = DAC 1 Select: DAC 1 mux select that allows us to switch between
    using the CRT (Off state) or TV (On State) filters on the board.

  - 13 = DAC 1 Alternate Load Detect: When the DAC 1 is not currently
    switched to a device that needs detection, this GPIO pin can be used
    to detect the alternate load on the green channel.

  - 14 = Stereo DAC Select: Chooses which DAC to use for the stereo
    goggles.

  - 15 = Stereo toggle: Switch between Left and Right eyes for the
    stereo goggles.

  - 16 = Thermal and External Power Detect: Sense bit when there’s a
    thermal event or the external power connector is connected or
    removed from the board.

  - 17 = Thermal Event Detect: Sense bit when there’s a thermal event
    sent from the thermal device.

  - 18 = Vtg rst: Input Signal from daughter card for Frame Lock
    interface headers.

  - 19 = Sus stat: Input requesting the suspend state be entered

  - 20 = Spread0: Bit 0 of output to control Spread Spectrum if the chip
    isn’t I2C controlled.

  - 21 = Spread1: Bit 1 of output to control Spread Spectrum if the chip
    isn’t I2C controlled.

  - 22 = VDS FrameID0 - Bit 0 of the frame ID when using Virtual Display
    Switching.

  - 23 = VDS FrameID1: Bit 1 of the frame ID when using Virtual Display
    Switching.

  - 24 = FBVDDQ Select: Selects between:

  - : ON state: High FBVDD/Q voltage (i.e. 1.8V)

  - : OFF State: Low FBVDD/Q voltage (i.e. 1.5V)

  - 25 = Customer: This function is here to be used by the OEM. It just
    reserves the GPIO so our software will know not to use it.

  - 26 = VSEL 3: Voltage Select Bit 3

  - 27 = VSEL Default - Allow switching from default voltage (1) to
    selected voltage (0).

  - 28 = Tuner

  - 29 = Current Share

  - 30 = Current Share Enable

  - 31 = LCD0 Self Test. LCD0 corresponds to the LCD0 defined in LCD ID
    field in Connector Table.

  - 32 = LCD0 Lamp Status. LCD0 corresponds to the LCD0 defined in LCD
    ID field in Connector Table.

  - 33 = LCD0 Brightness - Allow brightness adjustment via PWM. (Must
    have PWM set when this is selected.). LCD0 corresponds to the LCD0
    defined in LCD ID field in Connector Table.

  - 34 = Required Power Sense. Similar to 16, but without the thermal
    half.

  - 35 = OverTemp - This GPIO will assert when the GPU has reached some
    adjustable temperature threshold

  - 36 = HDTV Select: Allows selection of lines driven between SDTV -
    Off state, and HDTV - On State.

  - 37 = HDTV Alt-Detect: Allows detection of the connectors that are
    not selected by HDTV Select. That is, if HDTV Select is currently
    selecting SDTV, then this GPIO would allow us to detect the presence
    of the HDTV connection.

  - 38 = Reserved

  - 39 = Optional Power Sense. Similar to 16 and 34, but without the
    thermal half and not necessary for normal non-overclocked
    operation.1

  - 40 = DAC 0 Select: DAC 0 mux that allows us to switch between using
    the CRT (Off state) or TV (On State) filters on the board.

  - 41 = Framelock daughter-card interrupt

  - 42 = SW Performance Level Slowdown. When asserted, the SW will lower
    it’s performance level to the lowest state.

  - 43 = HW Slowdown Enable. On assertion HW will slowdown clocks
    (NVCLK, HOTCLK) using either \_EXT\_POWER, \_EXT\_ALERT or
    \_EXT\_OVERT settings (depends on GPIO configured: 12, 9 & 8
    respectively). Than SW will take over, limit GPU p-state to battery
    level and disable slowdown. On deassertion SW will reenable slowdown
    and remove p-state limit. System will continue running full clocks.

  - 44 = Disable Power Sense. If asserted, this GPIO will remove the
    power sense circuit from affecting HW Slowdown.

  - 45 = RSET HDTV Select. Allows selecting between SDTV, On State, and
    HDTV, Off State, RSET values during TV detection.

  - 46 = FBVREF Select: Selects between:
    
    <div class="ulist">
    
      - ON state: High FBVREF voltage (i.e. 70% FBVDDQ)
    
      - OFF state: Low FBVREF voltage (i.e. 50% FBVDDQ)
    
    </div>

  - 47 = Reserved

  - 48 = Generic Initialized: This GPIO is used, but does not have a
    specific function assigned to it or has a function defined
    elsewhere. System software should initialize this GPIO using the
    \_INIT values for the chip. This function should be specified when a
    GPIO needs to be set statically during initialization. This is
    different than function 25, which implies that the GPIO is not used
    by NVIDIA software.

  - 49 = Inquiry for HD over SD TV boot preference. Allows user to
    select whether to boot to SDTV or component output by default.

  - 50 = Digital Encoder Interrupt Enable: For Si1930uC, a GPIO will be
    set ON to trigger interrupt to Si1930uC to enable I2C communication.
    When I2C transactions to the Si1930uC are complete, the drivers will
    set this GPIO to OFF.

  - 51 = Selects I2C communications between either DDC or I2C

  - 52 = Thermal Alert: Interrupt input from external thermal device.
    Indicates that the device needs to be serviced.

  - 53 = Thermal Critical: Comparator-driven input from external thermal
    device. Indicates that a temperature is above a critical limit.

  - 54 = Reserved

  - 55 = Reserved

  - 56 = Reserved

  - 57 = Reserved

  - 58 = Reserved

  - 59 = Reserved

  - 60 = SCART Select: Allows selection of lines driven between SDTV
    (S-Video, Composite) and SDTV (SCART).

  - 61 = Fan Speed Sense. This GPIO will sense a fan’s tachometer output
    (on 4-wire fans). In the beginning, it will be more for sensing a
    stuck fan than determining speed. Later GPUs will be able to measure
    the fan’s speed internally from the GPIO.

  - 62 = Reserved

  - 63 = ExtSync0 - Used with external framelock with GSYNC products. It
    also could be used for raster lock.

  - 64 = SLI Raster Sync A: This signal is carried across the SLI bus to
    synchronize the RG between GPUs. This signal will always be set as
    Alternate.

  - 65 = SLI Raster Sync B: This signal is carried across the SLI bus to
    synchronize the RG between GPUs. This signal will always be set as
    Alternate. This signal is just the second GPIO that can be used for
    Raster sync from each GPU. It should only be defined when we have 2
    pin sets being used on one board to allow more than two GPUs to run
    in SLI mode. One will be used with one pin set for input and the
    other will be used with the other pin set for output.

  - 66 = Swap Ready In A: This signal, which is related to Fliplocking,
    is used to sense the state of the FET drain, which is pulled high
    and is connected to the Swap Ready pin on the Distributed Rendering
    connector.

  - 67 = Swap Ready Out: This signal, which is related to Fliplocking,
    is used to drive the gate of an external FET.

  - 68 = Reserved

  - 69 = SCART 0: Bit 0 of the SCART Aspect Ratio Field

  - 70 = SCART 1: Bit 1 of the SCART Aspect Ratio Field

  - GPIOs 69 and 70 define a 2 bit SCART Aspect Ratio Field. Here’s the
    possible values for the SCART Aspect Ratio Field:
    
    <div class="ulist">
    
      - 0 = 4:3(12V)
    
      - 1 = 16:9(6V)
    
      - 2 = Undefined
    
      - 3 = SCART inactive (0 V)
    
    </div>

  - 71 HD Dongle Strap 0: Bit 0 of the HD Dongle Strap Field

  - 72 HD Dongle Strap 1: Bit 1 of the HD Dongle Strap Field

</div>

<div class="admonitionblock">

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td><div class="title">
Note
</div></td>
<td>GPIOs 71 and 72 define a 2 bit HD Dongle Strap Field. These two bits index into an array found at the <a href="#_hdtv_translation_table">HDTV Translation Table</a> that will determine the default HD standard.</td>
</tr>
</tbody>
</table>

</div>

<div class="ulist">

  - 73 = Thermal Alert Output: Output signal that indicates to other
    board component(s) that the gpu’s internal temp has exceeded a
    certain threshold for a duration longer than a programmed interval.

  - 74 = DisplayPort to DVI dongle present A, when this GPIO asserts, we
    need to configure DisplayPort encoder to output TMDS signal.

  - 75 = DisplayPort to DVI dongle present B, when this GPIO asserts, we
    need to configure DisplayPort encoder to output TMDS signal.

  - 76 = Power Alert, when this GPIO asserts, the on-board power supply
    controller needs attention.

  - 77 = DAC 0 Load Detect: When the DAC 0 is not currently switched to
    a device that needs detection, this GPIO pin can be used to detect
    the alternate display’s load on the green channel.

  - 78 = Analogix Encoder External Reset: For Analogix encoder, a GPIO
    is used to control the RESET\# line.

  - 79 = I2C SCL Keeper Circuit Enable. See {{Bug|273429}}. Possible
    logical values are:
    
    <div class="ulist">
    
      - OFF state: Normal operation (do nothing)
    
      - ON state: Enable the hardware to detect slave-issued stretches
        on the SCL line and hold SCL low.
    
    </div>

  - 80 = DVI to DAC connector switch. This GPIO allows for DAC 0 (TV) to
    be selected to route to the DVI Connector when the GPIO is set to
    the logical OFF state. When the GPIO is set to logical ON state, DAC
    1 (CRT) will be routed to the DVI connector.

  - 81 = Hotplug C: 3rd Hotplug signal

  - 82 = Hotplug D: 4th Hotplug signal

  - 83 = DisplayPort to DVI dongle present C, when this GPIO asserts, we
    need to configure DisplayPort encoder to output TMDS signal.

  - 84 = DisplayPort to DVI dongle present D, when this GPIO asserts, we
    need to configure DisplayPort encoder to output TMDS signal.

  - 85 = Maxim Max6305 or compatible external reset controller. Enabled
    is Active Low so init value should be Active High \[No inversions\]

  - 86 = Active display LED to indicate the GPU with active display in
    SLI mode.

  - 87 = SPDIF input.

  - 88 = TOSLINK input.

  - 89 = SPDIF/TOSLINK Select. When GPIO is set LOW, SPDIF is selected.
    When GPIO is set HI, TOSLINK is selected.

  - 90 = DPAUX/I2C select A. When this GPIO is set to Logical ON state,
    DPAUX will be selected. Logical OFF state selects I2C.

  - 91 = DPAUX/I2C select B. When this GPIO is set to Logical ON state,
    DPAUX will be selected. Logical OFF state selects I2C.

  - 92 = DPAUX/I2C select C. When this GPIO is set to Logical ON state,
    DPAUX will be selected. Logical OFF state selects I2C.

  - 93 = DPAUX/I2C select D. When this GPIO is set to Logical ON state,
    DPAUX will be selected. Logical OFF state selects I2C.

  - 94 = Hotplug E: 5th Hotplug signal

  - 95 = Hotplug F: 6th Hotplug signal

  - 96 = Hotplug G: 7th Hotplug signal

  - 99 = GPIO External Device 1 Interrupt - Used to surface an interrupt
    from a GPIO external device

  - 106 = Switched Outputs: This GPIO is used by the [switched outputs
    table](#_switched_outputs_table). A switched outputs GPIO must be
    processed by the INIT\_GPIO\_ALL devinit opcode and set to its init
    state.

  - 107 = Customer Asyncronous Read/Write - Allows a customer to use the
    GPIO for whatever purpose they want.

  - 108 = Access to MXM 3.0 bus’s Direct GPIO0 (Pin 26). Once the system
    has the MXM structure/GPIO Device structure which defines usage of
    Direct GPIO0, this GPU’s GPIO is the physical pin to take on any
    enabling/detection/disabling function defined in the MXM Output
    Device data structure with MXM Direct GPIO0.

  - 109 = Access to MXM 3.0 bus’s Direct GPIO1 (Pin 28). Once the system
    has the MXM structure/GPIO Device structure which defines usage of
    Direct GPIO1, this GPU’s GPIO is the physical pin to take on any
    enabling/detection/disabling function defined in the MXM Output
    Device data structure with MXM Direct GPIO1.

  - 110 = Access to MXM 3.0 bus’s Direct GPIO2 (Pin 30). Once the system
    has the MXM structure/GPIO Device structure which defines usage of
    Direct GPIO2, this GPU’s GPIO is the physical pin to take on any
    enabling/detection/disabling function defined in the MXM Output
    Device data structure with MXM Direct GPIO2.

  - 111 = HW Only Slowdown Enable. On assertion HW will slowdown clocks
    (NVCLK, HOTCLK) using \_EXT\_POWER settings (use only with GPIO12).
    No software action will be taken. On deassertion HW will release
    clock slowdown.

  - 112 = Swap Ready In B: This signal, which is related to Fliplocking,
    is used to sense the state of the FET drain, which is pulled high
    and is connected to the Swap Ready pin on the Distributed Rendering
    connector.

  - 113 = Trigger condition for PMU: Can either be triggered by system
    notify bit set in SBIOS postbox command register or an error
    entering into deep-idle.

  - 114 = Reserved for Swap Ready Out B

  - 115 = VSEL4: Voltage Select Bit 4

  - 116 = VSEL5: Voltage Select Bit 5

  - 117 = VSEL6: Voltage Select Bit 6

  - 118 = VSEL7: Voltage Select Bit 7

  - 119 = LVDS Fast switch mux

  - 120 = Fan Failsafe PWM: The functionality controls FAN fail safe PWM
    generator. If function is present in VBIOS, GPIO should be
    configured as normal output and initially asserted. Once RM is
    loaded and FAN control is successfully initialized RM will dessert
    this pin to allow FAN\_PWM control.

  - 121 = External Power Emergency: This GPIO provides an input to let
    SW know when the GPU does not have enough power to initialize.

  - 122 = NVVDD PSI: The NVVDD Power State Indicator (PSI) signals the
    NVVDD power supply controller to switch to reduced phase operation
    (typically 1 or 2 phases) for efficiency in low power states. Here
    are the logical states:
    
    <div class="ulist">
    
      - ON state: Enable low power state (reduced phase operation)
    
      - OFF state: Disable low power state (all phase operation)
    
    </div>

  - 123 = Fan with Overtemp: denotes that the pin will be driven from
    PWM source that has capability to MAX duty cycle based on the
    thermal ALERT signal, as opposed to the already present "Fan"
    function which only outputs PWM. This PWM source is independent from
    the pwm source for "Fan" function.

  - 124 = POSTed GPU LED to indicate the GPU that was POSTed by the
    SBIOS.

  - 125 = Reserved

  - 126 = Reserved

  - 127 = Reserved

  - 128 = SMPBI Event Notification: Notifies the EC (or client of the
    SMBus Post Box Interface) of a pending GPU event requiring its
    attention.

  - 129 = PWM based Serial VID voltage control for NVVDD.

  - 130 = Reserved

  - 131 = SLI Bridge LED Brightness - Allow SLI Bridge brightness
    adjustment via PWM. (Must have PWM set when this is selected.)

  - 132 = Cover LOGO LED Brightness - Allow Cover LOGO brightness
    adjustment via PWM. (Must have PWM set when this is selected.)

  - 133 = Panel Self Refresh Frame Lock A : This function is defined for
    Self-Refresh Panel. The SR panel will send the frame-lock interrupt
    to GPU to sync the raster frame signal.

  - 134 = FB Clamp: This function is used to monitor the FB clamp signal
    driven by the Embedded Controller (EC) for JT memory self-refresh
    entry and exit.

  - 135 = FB Clamp Toggle Request: This function is used to request the
    Embedded Controller (EC) to toggle the FB clamp signal.

  - 136 = Reserved

  - 137 = Reserved

  - 138 = LCD1 backlight: Backlight control. LCD1 corresponds to the
    LCD1 defined in LCD ID field in Connector Table.

  - 139 = LCD1 power: Panel Power control. LCD1 corresponds to the LCD1
    defined in LCD ID field in Connector Table.

  - 140 = LCD1 Power Status: Panel Power status. LCD1 corresponds to the
    LCD1 defined in LCD ID field in Connector Table.

  - 141 = LCD1 Self Test. LCD1 corresponds to the LCD1 defined in LCD ID
    field in Connector Table.

  - 142 = LCD1 Lamp Status. LCD1 corresponds to the LCD1 defined in LCD
    ID field in Connector Table.

  - 143 = LCD1 Brightness - Allow brightness adjustment via PWM. (Must
    have PWM set when this is selected.). LCD1 corresponds to the LCD1
    defined in LCD ID field in Connector Table.

  - 144 = LCD2 backlight: Backlight control. LCD2 corresponds to the
    LCD2 defined in LCD ID field in Connector Table.

  - 145 = LCD2 power: Panel Power control. LCD2 corresponds to the LCD2
    defined in LCD ID field in Connector Table.

  - 146 = LCD2 Power Status: Panel Power status. LCD2 corresponds to the
    LCD2 defined in LCD ID field in Connector Table.

  - 147 = LCD2 Self Test. LCD2 corresponds to the LCD2 defined in LCD ID
    field in

  - Connector Table.

  - 148 = LCD2 Lamp Status. LCD2 corresponds to the LCD2 defined in LCD
    ID field in Connector Table.

  - 149 = LCD2 Brightness - Allow brightness adjustment via PWM. (Must
    have PWM set when this is selected.). LCD2 corresponds to the LCD2
    defined in LCD ID field in Connector Table.

  - 150 = LCD3 backlight: Backlight control. LCD3 corresponds to the
    LCD3 defined in LCD ID field in Connector Table.

  - 151 = LCD3 power: Panel Power control. LCD3 corresponds to the LCD3
    defined in LCD ID field in Connector Table.

  - 152 = LCD3 Power Status: Panel Power status. LCD3 corresponds to the
    LCD3 defined in LCD ID field in Connector Table.

  - 153 = LCD3 Self Test. LCD3 corresponds to the LCD3 defined in LCD ID
    field in Connector Table.

  - 154 = LCD3 Lamp Status. LCD3 corresponds to the LCD3 defined in LCD
    ID field in Connector Table.

  - 155 = LCD3 Brightness - Allow brightness adjustment via PWM. (Must
    have PWM set when this is selected.). LCD3 corresponds to the LCD3
    defined in LCD ID field in Connector Table.

  - 156 = LCD4 backlight: Backlight control. LCD4 corresponds to the
    LCD4 defined in LCD ID field in Connector Table.

  - 157 = LCD4 power: Panel Power control. LCD4 corresponds to the LCD4
    defined in LCD ID field in Connector Table.

  - 158 = LCD4 Power Status: Panel Power status. LCD4 corresponds to the
    LCD4 defined in LCD ID field in Connector Table.

  - 159 = LCD4 Self Test. LCD4 corresponds to the LCD4 defined in LCD ID
    field in Connector Table.

  - 160 = LCD4 Lamp Status. LCD4 corresponds to the LCD4 defined in LCD
    ID field in Connector Table.

  - 161 = LCD4 Brightness - Allow brightness adjustment via PWM. (Must
    have PWM set when this is selected.). LCD4 corresponds to the LCD4
    defined in LCD ID field in Connector Table.

  - 162 = LCD5 backlight: Backlight control. LCD5 corresponds to the
    LCD5 defined in LCD ID field in Connector Table.

  - 163 = LCD5 power: Panel Power control. LCD5 corresponds to the LCD5
    defined in LCD ID field in Connector Table.

  - 164 = LCD5 Power Status: Panel Power status. LCD5 corresponds to the
    LCD5 defined in LCD ID field in Connector Table.

  - 165 = LCD5 Self Test. LCD5 corresponds to the LCD5 defined in LCD ID
    field in Connector Table.

  - 166 = LCD5 Lamp Status. LCD5 corresponds to the LCD5 defined in LCD
    ID field in Connector Table.

  - 167 = LCD5 Brightness - Allow brightness adjustment via PWM. (Must
    have PWM set when this is selected.). LCD5 corresponds to the LCD5
    defined in LCD ID field in Connector Table.

  - 168 = LCD6 backlight: Backlight control. LCD6 corresponds to the
    LCD6 defined in LCD ID field in Connector Table.

  - 169 = LCD6 power: Panel Power control. LCD6 corresponds to the LCD6
    defined in LCD ID field in Connector Table.

  - 170 = LCD6 Power Status: Panel Power status. LCD6 corresponds to the
    LCD6 defined in LCD ID field in Connector Table.

  - 171 = LCD6 Self Test. LCD6 corresponds to the LCD6 defined in LCD ID
    field in Connector Table.

  - 172 = LCD6 Lamp Status. LCD6 corresponds to the LCD6 defined in LCD
    ID field in Connector Table.

  - 173 = LCD6 Brightness - Allow brightness adjustment via PWM. (Must
    have PWM set when this is selected.). LCD6 corresponds to the LCD6
    defined in LCD ID field in Connector Table.

  - 174 = LCD7 backlight: Backlight control. LCD7 corresponds to the
    LCD7 defined in LCD ID field in Connector Table.

  - 175 = LCD7 power: Panel Power control. LCD7 corresponds to the LCD7
    defined in LCD ID field in Connector Table.

  - 176 = LCD7 Power Status: Panel Power status. LCD7 corresponds to the
    LCD7 defined in LCD ID field in Connector Table.

  - 177 = LCD7 Self Test. LCD7 corresponds to the LCD7 defined in LCD ID
    field in Connector Table.

  - 178 = LCD7 Lamp Status. LCD7 corresponds to the LCD7 defined in LCD
    ID field in Connector Table.

  - 179 = LCD7 Brightness - Allow brightness adjustment via PWM. (Must
    have PWM set when this is selected.). LCD7 corresponds to the LCD7
    defined in LCD ID field in Connector Table.

  - 180 = Reserved

  - 255 = 0xFF = Skip Entry. This allows for quick removal of an entry
    from the GPIO Assignment table.

</div>

<div class="paragraph">

<div class="title">

Output HW select (23:16)

</div>

This field specifies *HW Select* value which has to be directly written
by software into the OUTPUT register field for deciding which output HW
function/enumerant will drive the PIN.

</div>

<div class="paragraph">

Values as specified in class20x HW manual

</div>

<div class="ulist">

  - 0x00 = SEL\_NORMAL

  - 0x40 = SEL\_RASTER\_SYNC\_0

  - 0x41 = SEL\_RASTER\_SYNC\_1

  - 0x42 = SEL\_RASTER\_SYNC\_2

  - 0x43 = SEL\_RASTER\_SYNC\_3

  - 0x48 = SEL\_STEREO\_0

  - 0x49 = SEL\_STEREO\_1

  - 0x4A = SEL\_STEREO\_2

  - 0x4B = SEL\_STEREO\_3

  - 0x50 = SEL\_SWAP\_READY\_OUT\_0

  - 0x51 = SEL\_SWAP\_READY\_OUT\_1

  - 0x52 = SEL\_SWAP\_READY\_OUT\_2

  - 0x53 = SEL\_SWAP\_READY\_OUT\_3

  - 0x58 = SEL\_THERMAL\_OVERT

  - 0x59 = SEL\_FAN\_ALERT

  - 0x5A = SEL\_THERMAL\_LOAD\_STEP\_0

  - 0x5B = SEL\_THERMAL\_LOAD\_STEP\_1

  - 0x5C = SEL\_PWM\_OUTPUT

  - 0x80 = SEL\_SOR0\_TMDS\_OUT\_PWM

  - 0x81 = SEL\_SOR0\_TMDS\_OUT\_PINA

  - 0x82 = SEL\_SOR0\_TMDS\_OUT\_PINB

  - 0x84 = SEL\_SOR1\_TMDS\_OUT\_PWM

  - 0x85 = SEL\_SOR1\_TMDS\_OUT\_PINA

  - 0x86 = SEL\_SOR1\_TMDS\_OUT\_PINB

  - 0x88 = SEL\_SOR2\_TMDS\_OUT\_PWM

  - 0x89 = SEL\_SOR2\_TMDS\_OUT\_PINA

  - 0x8A = SEL\_SOR2\_TMDS\_OUT\_PINB

  - 0x8C = SEL\_SOR3\_TMDS\_OUT\_PWM

  - 0x8D = SEL\_SOR3\_TMDS\_OUT\_PINA

  - 0x8E = SEL\_SOR3\_TMDS\_OUT\_PINB

</div>

<div class="paragraph">

<div class="title">

Input HW select (28:24)

</div>

This field specifies the input HW function *number* which needs to be
routed to the given pin (given by GPIO Number) which is also equivalent
to the index of the INPUT\_CNTL register that needs to be programmed.

</div>

<div class="paragraph">

Right now the manual specifies space for 24 functions(1-24) which are
given below. Note that 0 is not a valid input function in HW and is used
only to specify that no input function needs to be programmed on the
given pin.

</div>

<div class="ulist">

  - 00 / 0x00 = No Input function needs to be programmed on the given
    pin. Note that 0 is not a valid input value in HW.

  - 01 / 0x01 = NV\_PMGR\_GPIO\_INPUT\_FUNC\_AUX\_HPD(0)

  - 02 / 0x02 = NV\_PMGR\_GPIO\_INPUT\_FUNC\_AUX\_HPD(1)

  - 03 / 0x03 = NV\_PMGR\_GPIO\_INPUT\_FUNC\_AUX\_HPD(2)

  - 04 / 0x04 = NV\_PMGR\_GPIO\_INPUT\_FUNC\_AUX\_HPD(3)

  - 05 / 0x05 = NV\_PMGR\_GPIO\_INPUT\_FUNC\_AUX\_HPD(4)

  - 06 / 0x06 = NV\_PMGR\_GPIO\_INPUT\_FUNC\_AUX\_HPD(5)

  - 07 / 0x07 = NV\_PMGR\_GPIO\_INPUT\_FUNC\_AUX\_HPD(6)

  - 09 / 0x09 = NV\_PMGR\_GPIO\_INPUT\_FUNC\_RASTER\_SYNC(0)

  - 10 / 0x0A = NV\_PMGR\_GPIO\_INPUT\_FUNC\_RASTER\_SYNC(1)

  - 11 / 0x0B = NV\_PMGR\_GPIO\_INPUT\_FUNC\_RASTER\_SYNC(2)

  - 12 / 0x0C = NV\_PMGR\_GPIO\_INPUT\_FUNC\_RASTER\_SYNC(3)

  - 17 / 0x11 = NV\_PMGR\_GPIO\_INPUT\_FUNC\_SWAP\_READY(0)

  - 18 / 0x12 = NV\_PMGR\_GPIO\_INPUT\_FUNC\_SWAP\_READY(1)

  - 21 / 0x15 = NV\_PMGR\_GPIO\_INPUT\_FUNC\_THERMAL\_OVERTEMP

  - 22 / 0x16 = NV\_PMGR\_GPIO\_INPUT\_FUNC\_THERMAL\_ALERT

  - 23 / 0x17 = NV\_PMGR\_GPIO\_INPUT\_FUNC\_POWER\_ALERT

  - 24 / 0x18 = NV\_PMGR\_GPIO\_INPUT\_FUNC\_TACH

</div>

<div class="paragraph">

<div class="title">

GSYNC Header (29:29)

</div>

GSYNC Header Connection. Possible values are:

</div>

<div class="ulist">

  - 0 - Not Connected

  - 1 - Connected

</div>

<div class="paragraph">

RM is responsible for discerning Raster Sync or Flip Lock from the GPIO
Function.

</div>

<div class="paragraph">

<div class="title">

Pulse Width Modulate (PWM)

</div>

If this bit is 1, then this GPIO is used with PWM.

</div>

<div class="paragraph">

<div class="title">

Lock Pin Number (35:32)

</div>

The lock pin number associated with this entry. In ISO designs there are
currently four lock pins that are either assigned to GPIO pins or
internal dedicated pins. This only applies to a subset of GPIO
functions. Depending on the chip, some lock pins are done with real
GPIO’s so they have a real GPIO number and the I/O Type Field is set
to NV\_GPIO\_IO\_TYPE\_GPIO, while other lock pins do not have a real
GPIO so they are set to NV\_GPIO\_IO\_TYPE\_DEDICATED\_LOCK\_PIN and the
GPIO number is meaningless (but is always set to zero). This field must
be 0xF for GPIO functions that do not involve a lock pin.

</div>

<div class="paragraph">

<div class="title">

Off Data (FT) (36:36)

</div>

This field determines in what physcial data output must be present on
the GPIO pin to indicate the logical OFF signal. If this bit is 0, then
the software will set the GPIO pin to 0 when it wants to turn the
function off.

</div>

<div class="paragraph">

<div class="title">

Off Enable (FE) (37:37)

</div>

This field determines in which physical direction the GPIO should be
placed when requesting the logical function to be OFF. If this bit is 0,
then the GPIO will be set as an Output when OFF is requested. If this
bit is a 1, then the GPIO will be set as an Input when OFF is requested.

</div>

<div class="paragraph">

<div class="title">

On Data (OT) (38:38)

</div>

This field determines what physical data output must be present on the
GPIO pin to indicate the logical ON signal. If this bit is 0, then the
software will set the GPIO pin to 0 when it wants to turn the function
on.

</div>

<div class="paragraph">

<div class="title">

On Enable (OE) (39:39)

</div>

This field determines in which physical direction the GPIO should be
placed when requesting the logical function to be ON. If this bit is 0,
then the GPIO will be set as an Output when ON is requested. If this bit
is a 1, then the GPIO will be set as an Input when ON is requested.

</div>

<div class="openblock">

<div class="content">

<div class="paragraph">

Note:

</div>

<div class="paragraph">

Some GPIOs have some overloading with HW Slowdown features and the
detected presence of a thermal chip. HW Slowdown consists of two parts:

</div>

<div class="olist loweralpha">

1.  Enabling/Disabling the functionality through GPIO 8. (Note, this
    functionality is only available on NV18 and NV30+ chips.)

2.  Triggering GPIO 8 when the functionality is enabled. The trigger can
    come from a thermal device, external power connector, some logic on
    the board, a combination of the above, etc. The trigger method is
    what the GPIO function should define, but by defining it, we
    understand that we must enable HW slowdown (A) as well.

</div>

<div class="paragraph">

Trigger or Assert implies that the GPIO 8 is brought LOW and since the
functionality is enabled (A), the HW Clocks are reduced by 2x, 4x or 8x.
The opposite of trigger/assert is deassert.

</div>

<div class="paragraph">

In most cases today, HW Slowdown is set to ACTIVE LOW due to the ACTIVE
LOW signal from the thermal chips. We can program GPIO 8 based HW
Slowdown to be ACTIVE HIGH, but then the trigger level for the line
routed to GPIO 8 must follow the ACTIVE HIGH signaling.

</div>

<div class="paragraph">

Here is a list of all the different GPIO functions and their meaning in
relationship to the above:

</div>

<div class="ulist">

  - 16 = Thermal and External Power Detect: If attached to GPIO 8,
    assumes HW Slowdown enabled (A). If thermal device is not found, HW
    Slowdown is disabled (A). Here’s a logical diagram of this
    connection: <span class="image"> ![Thermal GPIO and Power
    routing](ThermalPower.gif) </span>

  - 17 = Thermal Event Detect: Same as above, but without the Power
    Connected signal. Specifically, the Thermal ASSERT is routed
    directly to GPIO 8.

  - 34 = Required Power Sense: This version is similar to Thermal and
    External Power Detect, but without the Thermal ASSERT signal.
    Specifically, the Power Connected signal is routed directly to GPIO
    8. The intention of the SW is to disable HW Slowdown (A) with this
    function.

  - 39 = Optional Power Sense: Same as Required Power Sense with regards
    to HW Slowdown.

  - 42 = SW Performance Level Slowdown: This GPIO function will act as a
    trigger point for the SW to lower the clocks. HW Slowdown (A) is not
    enabled.

  - 43 = HW Slowdown Enable: This function strictly allows for an
    undefined trigger point to cause HW Slowdown. There is no
    requirement to have a thermal device present in order to use HW
    Slowdown as in the functions Thermal and External Power Detect (16)
    and Thermal Event Detect (17).

  - 44 = Disable Power Sense. If asserted, this GPIO will remove the
    power sense circuit from affecting HW Slowdown. Note that HW
    Slowdown enable/disable (A) is not affected by the usage of this
    functionality. This function exists only to change the trigger
    method (B) for HW Slowdown. Here’s a logical diagram of this
    connection: <span class="image"> ![Thermal GPIO and Power with
    Disable routing](ThermalPowerDisable.gif) </span>

  - 52 = Thermal Alert and 53 = Thermal Critical. Although we have other
    thermal inputs that are tied to GPIO8, these can be assigned to any
    GPIO, and can cover many different situations.

  - 79 = The Analogix Encoder implements clock stretching in a manner
    that our SW emulated I2C cannot properly handle. To workaround this
    issue, a keeper circuit is added to detect slave issued stretches on
    the SCL and hold the SCL line. This allows our GPU to properly
    communicate with the Analogix chip. The keeper circuit is turned on
    and off at specific points during the I2C transaction.

</div>

</div>

</div>

<div class="admonitionblock">

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td><div class="title">
Note
</div></td>
<td>The presence of the GSYNC header can be positively determined by (1 == GSYNCRasterSync) for any non-skip entry or (1 == GSYNCFlipLock) for any non-skip entry.</td>
</tr>
</tbody>
</table>

</div>

<div class="sect3">

#### Lock Pins

<div class="paragraph">

There is a subset of GPIO functions that are "lock pins". In the case of
an entry that has one of these lock pin GPIO functions, the Lock Pin
Number Field tells which lock pin the functionality is mapped to.

</div>

<div class="paragraph">

Depending on the chip, some lock pins are done with real GPIOs so they
have a real GPIO number and the I/O Type Field is set to
NV\_GPIO\_IO\_TYPE\_GPIO, while other lock pins do not have a real GPIO
so they are set to NV\_GPIO\_IO\_TYPE\_DEDICATED\_LOCK\_PIN and the GPIO
number is meaningless (but is always set to zero).

</div>

<div class="paragraph">

Lockpins can be thought of as IO interface to the display HW. For
example; a head/rg can be programmed to be connected to a lockpin. The
lockpin can interface with GPIOs on the other side.

</div>

<div class="paragraph">

<span class="image"> ![Stall lock pin configuration](Stall_lock_dcb.jpg)
</span>

</div>

</div>

<div class="sect2">

### External GPIO Assignment Master Table

<div class="paragraph">

Some boards require extra control, since we don’t have enough internal
GPIO pins to manage them. The board designers add an external chip that
is used to control more GPIO pins on the board. Because we expect that
there could be more than just one external GPIO controller on the board,
we have separated the tables into Master and Specific. The Master table
lists pointers to all the different external GPIO controllers on the
board. The Specific Table lists the data associated with one controller
on the board. A pointer to the External GPIO Assignment Master Table is
found in the GPIO Assignment Table Header.

</div>

<div class="paragraph">

The Master Table is made up of two parts: the Header and the Entries.
The Entries follow immediately after the Header.

</div>

<div class="sect3">

#### External GPIO Assignment Master Table Header

<div class="tableblock">

31

</div>

</div>

</div>

30

29

28

27

26

25

24

23

22

21

20

19

18

17

16

15

14

13

12

11

10

9

8

7

6

5

4

3

2

1

0

Entry Size = 2

Entry Count

Header Size = 4

Version = 0x40

<div class="paragraph">

<div class="title">

Version

</div>

Version number of the GPIO Assignment Master Table Header and Entries.
The GPIO Assignment Master Table version will start with 4.0, or a value
of 0x40, in this field. If this version is 0, then the driver will
assume that this table is invalid.

</div>

<div class="paragraph">

<div class="title">

Header Size

</div>

Size of the GPIO Assignment Master Table Header in bytes. Initially,
this is 4 bytes.

</div>

<div class="paragraph">

<div class="title">

Entry Count

</div>

Number of GPIO Assignment Table Entries starting directly after the end
of this header.

</div>

<div class="paragraph">

<div class="title">

Entry Size

</div>

Size of each Master Table Entry in bytes. Initially, this is 2 bytes.

</div>

<div class="sect3">

#### External GPIO Assignment Master Table Entry

<div class="tableblock">

|                      Name                       | Bit width | Values and Meaning                                                                           |
| :---------------------------------------------: | :-------- | :------------------------------------------------------------------------------------------- |
| External GPIO Assignment Specific Table Pointer | 16        | Pointer to an External GPIO Assignment Specific Table. A value of 0 here means *skip entry*. |

</div>

</div>

<div class="sect2">

### External GPIO Assignment Specific Table

<div class="paragraph">

The Specific Table is made up of two parts, the Header and the Entries.
The Entries follow immediately after the Header.

</div>

<div class="sect3">

#### External GPIO Assignment Specific Table Header

<div class="tableblock">

31

</div>

</div>

</div>

30

29

28

27

26

25

24

23

22

21

20

19

18

17

16

15

14

13

12

11

10

9

8

7

6

5

4

3

2

1

0

Entry Size = 4

Entry Count

Header Size = 7

Version = 0x40

<div class="tableblock">

55

</div>

54

53

52

51

50

49

48

47

46

45

44

43

42

41

40

39

38

37

36

35

34

33

32

Rsvd

P

Rsvd

xInt

I2C Address

External Type

<div class="paragraph">

<div class="title">

Version

</div>

Version number of the GPIO Assignment Specific Table Header and Entries.
The GPIO Assignment Master Table will version will start with 4.0, or a
value of 0x40, in this field. If this version is 0, then the driver will
assume that this table is invalid.

</div>

<div class="paragraph">

<div class="title">

Header Size

</div>

Size of the GPIO Assignment Specific Table Header in bytes. Initially,
this is 7 bytes.

</div>

<div class="paragraph">

<div class="title">

Entry Count

</div>

Number of GPIO Assignment Specific Table Entries starting directly after
the end of this table.

</div>

<div class="paragraph">

<div class="title">

Entry Size

</div>

Size of each Specific Table Entry in bytes. Initially, this is 4 bytes.

</div>

<div class="paragraph">

<div class="title">

External Type

</div>

The actual chip used to control the GPIO pins. Possible values are:

</div>

<div class="ulist">

  - 0: Unknown - Used to signify to skip an entire Specific Table.

  - 1: PCA9555 for 10-pin Personal Cinema VIVO pods

  - 2: ADT7473 Automatic Fan Controller Chip

  - 3: CX25875 General Purpose Output pins

  - 4: PCA9555 for GPIO pins on MXM external HDMI control

  - 5: PCA9536 for GPIO pins for HDMI/DVI Multiplexing

  - 6: PCA9555 for GPIOs

  - 7: PCA9536 for GPIOs

  - 8: PCA9555 for Napoleon

  - 9: ANX9805 for GPIOs

  - A: Pic18f24k20 GPIO expander

</div>

<div class="paragraph">

<div class="title">

I2C Address

</div>

7-bit I2C communication Address left shifted to bits 7:1, with a 0 in
bit 0. This is the standard I2C address specification for SW.

</div>

<div class="paragraph">

<div class="title">

External Device Interrupt Number (xInt)

</div>

This field gives the number of the external interrupt pin that is used
to signal interrupt requests by this device. Possible values are:

</div>

<div class="ulist">

  - 0: No interrupts will be generated by this device

  - 1: The function "GPIO Expansion 1 Interrupt" from the GPIO
    Assignment table will signal interrupts for this device

  - 2: reserved for future use

  - 3: reserved for future use

</div>

<div class="paragraph">

<div class="title">

External Communications Port (P)

</div>

This field defines which communications port is used for this device.
See the I2C Control Block Header for the listing of the Primary and
Secondary Communication ports.

</div>

<div class="sect3">

#### External GPIO Assignment Specific Table Entry

<div class="paragraph">

Each entry here is defined exactly like the internal GPIO Entries. The
only differences are:

</div>

<div class="ulist">

  - The GPIO number associated here. If the External GPIO labels its
    GPIOs with numbers, we will use their number labels for the GPIO
    number here. If the external chip does not label with numbers, but
    labels them like "GPIO ABC", then we’ll use the GPIO pin that is
    closest to pin 0 of the part as GPIO 0.

  - The GPIO functions associated here. All functions must be explicitly
    defined for any given External Type. They can be defined exactly
    like the internal internal GPIO Entries, but it must be explicitly
    defined that way. A complete list of function code definitions for
    each defined External Type will be added to this document, as they
    are defined.

  - The SKIP ENTRY is defined by the value 0 for all External GPIO Types
    instead of 63.

</div>

</div>

<div class="sect3">

#### GPIO Entries for External Type 1 - PCA9555 for 10-pin Personal Cinema VIVO pods

<div class="paragraph">

For this particular External Type ("1: PCA9555 for 10-pin Personal
Cinema VIVO pods") there is a physical limit of 16 GPIO pins.

</div>

<div class="paragraph">

Here are the functions as listed for External Type 1

</div>

<div class="ulist">

  - 0 = SKIP ENTRY: This allows for quick removal of an entry from the
    GPIO Assignment table.

  - 1 = DTERM\_LINE1A: used to control Japanese HDTV sets.

  - 2 = CONFIG\_480p576p: indicates whether the user desires 480p/576p
    support

  - 3 = DTERM\_LINE1B: used to control Japanese HDTV sets.

  - 4 = CONFIG\_720p: indicates whether the user desires 720p support

  - 5 = DTERM\_LINE2A: used to control Japanese HDTV sets.

  - 6 = CONFIG\_1080i: indicates whether the user desires 1080i support

  - 7 = DTERM\_LINE2B: used to control Japanese HDTV sets.

  - 8 = DTERM\_LINE3A: used to control Japanese HDTV sets.

  - 9 = POD\_LOAD\_DET: used to detect connections to SDTV connectors

  - 10 = DTERM\_LINE3B: used to control Japanese HDTV sets.

  - 11 = POD\_SEL\_2ND\_DEV: used to activate SDTV connectors

  - 12 = DTERM\_SENSE: used to detect connections to Japanese HDTV
    connectors

  - 13 = CONFIG\_SDTV\_NOT\_COMPONENT:: indicates whether the user
    prefers SDTV or component output as the boot default.

  - 14 = POD\_LOCALE\_BIT0: used to indicate the geopolitical locale of
    the POD design. See interpretation below.

  - 15 = POD\_LOCALE\_BIT1: used to indicate the geopolitical locale of
    the POD design. See interpretation below.

</div>

<div class="paragraph">

The locale bits are interpreted with this table

</div>

<div class="tableblock">

|   |   |                             |
| :- | :- | :-------------------------- |
| 0 | 0 | North America YPrPb POD     |
| 1 | 0 | Japanese D-Connector POD    |
| 0 | 1 | European SCART with RGB POD |
| 1 | 1 | Reserved                    |

</div>

</div>

<div class="sect3">

#### GPIO Entries for External Type 2 - ADT7473 Automatic Fan Controller Chip

<div class="paragraph">

Currently, there will be only one GPIO defined for this chip.

</div>

<div class="ulist">

  - 0 = SKIP ENTRY: This allows for quick removal of an entry from the
    GPIO Assignment table.

  - 1 = FANCONTROL: This GPIO will provide on, off, or on with PWM
    control. In addition, when set as an input, the fan controller will
    switch to automatic temperature-based fan control.

</div>

<div class="paragraph">

There are 3 physical fan controllers on this chip. To reference any of
these, use the GPIO Number to differentiate each controller.

</div>

</div>

<div class="sect3">

#### GPIO Entries for External Type 3 - CX25875 General Purpose Output pins

<div class="paragraph">

There are 3 physical GPIO pins on this chip. Additional GPIO
functionality may be added in a future revision.

</div>

<div class="ulist">

  - 0 = SKIP ENTRY: This allows for quick removal of an entry from the
    GPIO Assignment table.

  - 1 = SCART\_RGB: Used to control the TV output as Composite (low) or
    RGB format (high).

  - 2 = SCART\_VIDEO\_ASPECT: used to control ouput picture as 16x9
    (low) or 4x3 (high).

</div>

</div>

<div class="sect3">

#### GPIO Entries for External Type 4 - PCA9555 for GPIO pins on MXM external HDMI

<div class="ulist">

  - 0 = SKIP ENTRY: This allows for quick removal of an entry from the
    GPIO Assignment table.

  - 1 = Digital Encoder Interrupt Enable: used to control I2C CLK line
    for SI1930 firmware update.

  - 2 = si1930uC Programming: used to control SI1930 firmware update.

  - 3 = si1930uC Reset: used to control reset signal of SI1930 uC.

</div>

</div>

<div class="sect3">

#### GPIO Entries for External Type 5 - PCA9536 for GPIO pins for HDMI/DVI Multiplexing

<div class="ulist">

  - 0 = SKIP ENTRY: This allows for quick removal of an entry from the
    GPIO Assignment table.

  - 1 = DVI/HDMI Select: controls whether the display data is routed to
    the DVI device or to the HDMI device.

  - 2 = I2C HDMI Enable: enables or disables the I2C bus for the HDMI
    device.

  - 3 = I2C DVI Enable: enables or disables the I2C bus for the DVI
    device.

</div>

<div class="paragraph">

Note that HDMI and DVI enable are mutually exclusive and may never be
asserted at the same time. See table below for additional information.

</div>

<div class="tableblock">

|           |                 |                 |                |
| :-------- | :-------------- | :-------------- | :------------- |
|           | DVI/HDMI Select | I2C HDMI Enable | I2C DVI Enable |
| DVI-Mode  | 1               | 0               | 1              |
| HDMI-Mode | 0               | 1               | 0              |

</div>

</div>

<div class="sect3">

#### GPIO Entries for External Type 6 and 7 - PCA9555 and PCA9536 for GPIOs

<div class="paragraph">

This define was originally defined to support MXM, but has more general
applicability.

</div>

<div class="ulist">

  - 0 = SKIP ENTRY: This allows for quick removal of an entry from the
    GPIO Assignment table.

  - 1 = Output Device Control: Used for DDC Bus Expander or Mux control
    (Switched Outputs)

  - 5 = Japanese D connector line 1

  - 6 = Japanese D connector line 2

  - 7 = Japanese D connector line 3

  - 8 = Japanese D connector plug insertion detect

  - 9 = Japanese D connector spare line 1

  - 10 = Japanese D connector spare line 2

  - 11 = Japanese D connector spare line 3

  - 12 = VSEL0: Voltage Select Bit 0

  - 13 = VSEL1: Voltage Select Bit 1

  - 14 = VSEL2: Voltage Select Bit 2

  - 15 = VSEL3: Voltage Select Bit 3

  - 16 = VSEL4: Voltage Select Bit 4

  - 17 = VSEL5: Voltage Select Bit 5

  - 18 = VSEL6: Voltage Select Bit 6

  - 19 = VSEL7: Voltage Select Bit 7

  - 31 = LCD Self Test

  - 32 = LCD Lamp Status

  - 36 = HDTV Select: Allows selection of lines driven between SDTV (OFF
    state) and HDTV (ON state)

  - 37 = HDTV Alt-Detect: Allows detection of the connectors that are
    not selected by HDTV Select. That is, if HDTV Select is currently
    selecting SDTV, then this GPIO would allow us detect the presence of
    the HDTV connection.

</div>

</div>

<div class="sect3">

#### GPIO Entries for External Type 8 - PCA9555 for S/PDif Detect and TV resolution LEDs

<div class="ulist">

  - 0 = SKIP ENTRY: This allows for quick removal of an entry from the
    GPIO Assignment table.

  - 1 = LED for 480/576i (Output)

  - 2 = LED for 480/576p (Output)

  - 3 = LED for 720p (Output)

  - 4 = LED for 1080i (Output)

  - 5 = LED for 1080p (Output)

  - 6 = HDAudio Signal Detect (Input)

  - 7 = S/PDif 0 (Coax) Signal Detect (Input)

  - 8 = S/PDif 1 (Header) Signal Detect (Input)

  - 9 = S/PDif Input Select - 0. Coax, 1. Header (Output)

  - 10 = Panic Button - Resets screen resolution to the lowest possible
    setting (Input)

  - 11 = Resolution Change Button - Changes the screen resolution to its
    next highest setting (Input)

</div>

</div>

<div class="sect3">

#### GPIO Entries for External Type 9 - ANX9805 External DP Encoder GPIO (deprecated)

<div class="admonitionblock">

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td><div class="title">
Note
</div></td>
<td>ANX9805 is deprecated on Fermi+</td>
</tr>
</tbody>
</table>

</div>

<div class="ulist">

  - 0 = SKIP ENTRY: This allows for quick removal of an entry from the
    GPIO Assignment table.

  - 1 = DP2DVI Dongle A: This GPIO is used to detect DP2DVI dongle’s
    presence (input) and is associated with the Connector Table’s DP2DVI
    A bit.

  - 2 = DP2DVI Dongle B: This GPIO is used to detect DP2DVI dongle’s
    presence (input) and is associated with the Connector Table’s DP2DVI
    B bit.

  - 3 = DP2DVI Dongle C: This GPIO is used to detect DP2DVI dongle’s
    presence (input) and is associated with the Connector Table’s DP2DVI
    C bit.

  - 4 = DP2DVI Dongle D: This GPIO is used to detect DP2DVI dongle’s
    presence (input) and is associated with the Connector Table’s DP2DVI
    D bit.

</div>

</div>

<div class="sect3">

#### GPIO Entries for External Type A Pic18f24k20 GPIO expander for P678/668 (deprecated)

<div class="admonitionblock">

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td><div class="title">
Note
</div></td>
<td>Pic18f24k20 GPIO expander is deprecated on Fermi+</td>
</tr>
</tbody>
</table>

</div>

<div class="ulist">

  - 0 = SKIP ENTRY: This allows for quick removal of an entry from the
    GPIO Assignment table.

  - 1 = Output Device Control: Used for DDC Bus Expander or Mux control
    (Switched Outputs).

</div>

</div>

<div class="sect1">

## Spread Spectrum Table

<div class="sectionbody">

<div class="paragraph">

This table is not required in the ROM. This table only needs to be
defined if the specific board requires spread spectrum. This table will
be used by both the VBIOS and the driver.

</div>

<div class="sect2">

### Spread Spectrum Table Header

<div class="tableblock">

|    Name     | Bit width | Values and Meaning                                                                                                                                                                                                                                              |
| :---------: | :-------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   Version   | 8         | Version \# of the Spread Spectrum Table Header and Entries. The current Spread Spectrum Table is version with 4.1, a value of 0x41 for this field. If the version is 0, then this table will be considered invalid and the driver will not use spread spectrum. |
| Header Size | 8         | Size of the Spread Spectrum Table Header in Bytes. Version 4.1 starts with 5 bytes.                                                                                                                                                                             |
| Entry Count | 8         | Number of Spread Spectrum Table Entries starting directly after the end of this table.                                                                                                                                                                          |
| Entry Size  | 8         | Size of Each Entry in bytes. Version 4.1 are currently 2 bytes each.                                                                                                                                                                                            |
|    Flags    | 8         | Flags for Spread Spectrum, currently unused. All bits are reserved and set to 0.                                                                                                                                                                                |

</div>

</div>

<div class="sect2">

### Spread Spectrum Table Entry

<div class="tableblock">

15

</div>

</div>

</div>

</div>

14

13

12

11

10

9

8

7

6

5

4

3

2

1

0

R

T

FreqDt

Indx

R

VS

V

<div class="tableblock">

|  Name  | Bit width | Values and Meaning                        |
| :----: | :-------- | :---------------------------------------- |
|   V    | 1         | Set if this is a valid entry              |
|   VS   | 2         | VPLL spread Source                        |
|   R    | 1         | Reserved, set as 0                        |
|  Indx  | 4         | DCB Index                                 |
| FreqDt | 6         | Frequency Delta in 0.05% units            |
|   T    | 1         | Spread profile type, 0 = center, 1 = down |
|   R    | 1         | Reserved, set as 0                        |

</div>

<div class="paragraph">

<div class="title">

Valid

</div>

This field defines whether this entry is valid or not. Defined values
are:

</div>

<div class="ulist">

  - 0 = Entry is invalid and should be skipped.

  - 1 = Entry is valid.

</div>

<div class="paragraph">

<div class="title">

VPLL Source

</div>

This field lists the source of the VPLL spread. Defined values are:

</div>

<div class="ulist">

  - 0 = Reference, GPU Internal Source 0 (INTERNAL\_SPREAD\_0)

  - 1 = Reference, GPU Internal Source 1 (INTERNAL\_SPREAD\_1)

  - 2 = Reference, GPU External Source (EXTERNAL\_SPREAD)

  - 3 = Self, PLL Internal Mechanism

</div>

<div class="paragraph">

<div class="title">

DCB Index

</div>

This field lists the associated DCB Index device that should enable
spread on VPLL while in use.

</div>

<div class="paragraph">

<div class="title">

Frequency Delta

</div>

Delta from target frequency (0.05%).

</div>

<div class="paragraph">

<div class="title">

Spread Type

</div>

Spread profile type. Defined values are:

</div>

<div class="ulist">

  - 0 = Center Spread

  - 1 = Down Spread

</div>

<div class="paragraph">

<div class="title">

Notes

</div>

The Frequency Delta and Type fields inside the Entry above are only used
when *VPLL Source* is set to 3 (i.e., Self, PLL Internal Mechanism).
When calculating the configuration for the VPLL’s own spread, *Frequency
Delta* should be interpreted as delta from target frequency such that
center spread has a bandwidth of

</div>

<div class="literalblock">

<div class="content">

``` 
  (2 x SpreadSpectrumTableEntry.FrequencyDelta)
```

</div>

</div>

<div class="paragraph">

and down spread has a bandwidth of

</div>

<div class="literalblock">

<div class="content">

``` 
  (1 x SpreadSpectrumTableEntry.FrequencyDelta)
```

</div>

</div>

<div class="paragraph">

The target modulation frequency is assumed to be 33 kHz.

</div>

<div class="sect1">

## I2C Device Table

<div class="sectionbody">

<div class="paragraph">

This table is not required in the ROM. This table only needs to be
defined if the board requires some specific driver handling of an I2C
device. This table will be used only by the the driver.

</div>

<div class="paragraph">

Specifically, this table grew from the need to define various new I2C HW
monitoring devices as well as HDTV chips.

</div>

<div class="sect2">

### I2C Device Table Header

<div class="tableblock">

31

</div>

</div>

</div>

</div>

30

29

28

27

26

25

24

23

22

21

20

19

18

17

16

15

14

13

12

11

10

9

8

7

6

5

4

3

2

1

0

Entry Size = 4

Entry Count

Header Size = 5

Version = 0x40

<div class="tableblock">

39

</div>

38

37

36

35

34

33

32

Flags

<div class="paragraph">

<div class="title">

Version

</div>

Version \# of the I2C Device Table Header and Entries. The version will
start with 4.0, a value of 0x40 here. If this version is 0, then the
driver will consider this table as invalid and will not use any of the
data present here.

</div>

<div class="paragraph">

<div class="title">

Header Size

</div>

Size of the I2C Device Table Header in Bytes. Initially, this is 5
bytes.

</div>

<div class="paragraph">

<div class="title">

Entry Count

</div>

Number of I2C Device Table Entries starting directly after the end of
this table.

</div>

<div class="paragraph">

<div class="title">

Entry Size

</div>

Size of Each Entry in bytes. Version 4.0 starts with 4 bytes.

</div>

<div class="paragraph">

<div class="title">

Flags

</div>

Flags for I2C Devices.

</div>

<div class="paragraph">

Currently defined fields are:

</div>

<div class="ulist">

  - Bit 0 : Disable External Device Probing: The driver spends some time
    probing for external devices like the framelock, SDI boards, or
    Thermal devices not found in the thermal tables. This bit is added
    to notify the driver that probing isn’t required because the board
    doesn’t support it. If set to 0, probing will still occur as normal.
    If set to 1, it will disable the probing on the board.

  - Bits 1-7 : Reserved. Set as 0.

</div>

<div class="sect2">

### I2C Device Table Header Version 4.0 Prior Sizes

<div class="tableblock">

|   DATE   | New Size | Last Inclusive Field |
| :------: | :------: | :------------------: |
|  Start   | 4 Bytes  |      Entry Size      |
| 09-14-06 | 5 Bytes  |        Flags         |

</div>

</div>

<div class="sect2">

### I2C Device Table Entry

<div class="tableblock">

31

</div>

</div>

30

29

28

27

26

25

24

23

22

21

20

19

18

17

16

15

14

13

12

11

10

9

8

7

6

5

4

3

2

1

0

Rsvd

RA

WA

P

Rsvd

I2C Address

Type

<div class="tableblock">

|    Name     | Bit width | Values and Meaning                 |
| :---------: | :-------- | :--------------------------------- |
|    Type     | 8         | Device (chip) type                 |
| I2C Address | 8         | 8 bit adjusted I2C address (LSB 0) |
|    Rsvd     | 4         | Reserved, set to 0                 |
|      P      | 1         | External Communications Port       |
|     WA      | 3         | Write Access privilege level       |
|     RA      | 3         | Read Access privilege level        |
|    Rsvd     | 5         | Reserved, set to 0                 |

</div>

<div class="paragraph">

<div class="title">

Type

</div>

Currently defined values are:

</div>

<div class="ulist">

  - THERMAL CHIPS
    
    <div class="ulist">
    
      - 0x01 = ADM 1032
    
      - 0x02 = MAX 6649
    
      - 0x03 = LM99
    
      - 0x06 = MAX 1617
    
      - 0x07 = LM64
    
      - 0x0A = ADT7473
    
      - 0x0B = LM89
    
      - 0x0C = TMP411
    
      - 0x0D = ADT7461
    
      - 0x04, 0x05, 0x08, and 0x09 = deprecated.
    
    </div>

  - I2C ANALOG TO DIGITAL CONVERTERS
    
    <div class="ulist">
    
      - 0x30 = ADS1112
    
    </div>

  - I2C POWER CONTROLLERS
    
    <div class="ulist">
    
      - 0xC0 = PIC16F690 micro controller (deprecated on Fermi+)
    
      - 0x40 = VT1103
    
      - 0x41 = PX3540 Primarion PX3540 Digital Multiphase PWM Voltage
        Controller
    
      - 0x42 = Volterra VT1165
    
      - 0x43 = CHiL CHL8203/8212/8213/8214
    
      - 0x44 = NCP4208
    
    </div>

  - SMBUS POWER CONTROLLERS
    
    <div class="ulist">
    
      - 0x48 = CHiL CHL8112A/B, CHL8225/8228
    
      - 0x49 = CHiL CHL8266, CHL8316
    
      - 0x4A = DS4424N
    
      - 0x4B = NCT3933U
    
    </div>

  - POWER SENSORS
    
    <div class="ulist">
    
      - 0x4C = INA219
    
      - 0x4D = INA209
    
      - 0x4E = INA3221
    
    </div>

  - 1 CLOCK GENERATORS
    
    <div class="ulist">
    
      - 0x50 = Cypress CY2XP304
    
    </div>

  - GENERAL PURPOSE GPIO CONTROLLERS
    
    <div class="ulist">
    
      - 0x60 = Philips PCA9555 device for EIAJ-4120 - Japanese HDTV
        support
    
      - 0x82 = Texas Instruments PCA9536 device for general-purpose
        remote I/O expansion
    
    </div>

  - FAN CONTROLS
    
    <div class="ulist">
    
      - 0x70 = ADT7473, dBCool Fan Controller
    
      - 0x71 = Reserved
    
      - 0x72 = Reserved
    
    </div>

  - HDMI COMPOSITOR/CONVERTER DEVICES
    
    <div class="ulist">
    
      - 0x80 = Silicon Image Microcontroller SI1930uC device for HDMI
        Compositor/Converter
    
    </div>

  - GPU I2CS CONTROLLERS
    
    <div class="ulist">
    
      - 0xB0 = GT21X - GF10X I2CS interface
    
      - 0xB1 = GF11X and beyond I2CS interface
    
    </div>

  - DISPLAY ENCODER TYPES
    
    <div class="ulist">
    
      - 0xD0 = Anx9805
    
    </div>

  - 0xFF = Skip Entry. This allows for quick removal of an entry from
    the I2C Devices Table.

</div>

<div class="paragraph">

<div class="title">

I2C Address

</div>

8-bit aligned, right shifted 7-bit address of the I2C device. The I2C
spec defines 7 bits for the address \[7:1\] of the device with 1 bit for
R/W \[0:0\]. So, generally, most addresses are listed in their 8 bit
adjusted form with 0 for the R/W bit. This field must list that 8-bit
adjusted address.

</div>

<div class="paragraph">

<div class="title">

External Communications Port

</div>

This field defines which communications port is used for this device.
See the I2C Control Block Header for the listing of the Primary and
Secondary Communication ports.

</div>

<div class="paragraph">

<div class="title">

Write Access

</div>

This field defines the write access privileges to specific levels.

</div>

<div class="paragraph">

Currently defined values are: \* 0x0-0x7 = Reserved

</div>

<div class="paragraph">

<div class="title">

Read Access

</div>

This field defines the read access privileges to specific levels.

</div>

<div class="paragraph">

Currently defined values are:

</div>

<div class="ulist">

  - 0x0-0x7 = Reserved

</div>

<div class="sect1">

## Connector Table

<div class="sectionbody">

<div class="paragraph">

This table is required in the ROM. This table should always be defined
to allow graphical representations of the board to be created. This
table will be used only by the the driver.

</div>

<div class="paragraph">

For purposes of this table a connector is defined as the end point on
the display path where one display can be attached. This may be the card
edge or attachment points on a breakout cable.

</div>

<div class="paragraph">

A connector can only output one stream at a time. So, if you have a
Low-Force Helix (LFH) port on the back of the card, the connector is
defined as a DVI-I adapter of that breakout cable. That is, there are 2
connectors for every 1 LFH port on the back of a card.

</div>

<div class="sect2">

### Connector Table Header

<div class="tableblock">

31

</div>

</div>

</div>

</div>

30

29

28

27

26

25

24

23

22

21

20

19

18

17

16

15

14

13

12

11

10

9

8

7

6

5

4

3

2

1

0

Entry Size = 4

Entry Count

Header Size = 5

Version = 0x40

<div class="tableblock">

39

</div>

38

37

36

35

34

33

32

Platform

<div class="paragraph">

<div class="title">

Version

</div>

Version \# of the Connector Table Header and Entries. The Version will
start with 4.0, a value of 0x40 here. If this version is 0, then the
driver will consider this table as invalid and will not use any of the
data present here.

</div>

<div class="paragraph">

<div class="title">

Header Size

</div>

Size of the Connector Table Header in bytes. Initially, this is 5 bytes.

</div>

<div class="paragraph">

<div class="title">

Entry Count

</div>

Number of Connector Table Entries starting directly after the end of
this table header.

</div>

<div class="paragraph">

<div class="title">

Entry Size

</div>

Size of Each Entry in bytes. Currently 4 bytes.

</div>

<div class="tableblock">

|    DATE    | New Size |
| :--------: | :------: |
|   Start    | 2 Bytes  |
| 2007-06-19 | 4 Bytes  |

</div>

<div class="paragraph">

<div class="title">

Platform

</div>

This field specifies the layout of the connectors.

</div>

<div class="paragraph">

Currently defined values are:

</div>

<div class="ulist">

  - 0x00 = Normal Add-in Card

  - 0x01 = Two back plate Add-in Cards (Used for tall fan sinks that
    cause adjacent PCI connection to be unusable)

  - 0x02 = Add-in card (Configurable) - All I2C ports need to be
    rescanned at boot for possible external device changes.

  - 0x07 = Desktop with Integrated full DP

  - 0x08 = Mobile Add-in Card. Generally have LVDS-SPWG connector on the
    north edge of the card away from the AGP/PCI bus.

  - 0x09 = MXM module

  - 0x10 = Mobile system with all displays on the back of the system.

  - 0x11 = Mobile system with display connectors on the back and left of
    the system.

  - 0x18 = Mobile system with extra connectors on the dock

  - 0x20 = Crush (nForce chipset) normal back plate design

</div>

<div class="sect2">

### Connector Table Entry

<div class="tableblock">

31

</div>

</div>

30

29

28

27

26

25

24

23

22

21

20

19

18

17

16

15

14

13

12

11

10

9

8

7

6

5

4

3

2

1

0

Rsvd

LCD ID

SRA

G

F

E

DID

DIC

DIB

DIA

DPD

DPC

D

C

DPB

DPA

B

A

Location

Type

<div class="tableblock">

|   Name   | Bit width | Values and Meaning                        |
| :------: | :-------- | :---------------------------------------- |
|   Type   | 8         | Connector Type                            |
| Location | 4         | Physical location description             |
|    A     | 1         | Hotplug A interrupt generation            |
|    B     | 1         | Hotplug B interrupt                       |
|   DPA    | 1         | DP2DVI A                                  |
|   DPB    | 1         | DP2DVI B                                  |
|    C     | 1         | Hotplug C interrupt                       |
|    D     | 1         | Hotplug D interrupt                       |
|   DPC    | 1         | DP2DVI C                                  |
|   DPD    | 1         | DP2DVI D                                  |
|   DIA    | 1         | DPAux/I2C-A                               |
|   DIB    | 1         | DPAux/I2C-B                               |
|   DIC    | 1         | DPAux/I2C-C                               |
|   DID    | 1         | DPAux/I2C-D                               |
|    E     | 1         | Hotplug E interrupt                       |
|    F     | 1         | Hotplug F interrupt                       |
|    G     | 1         | Hotplug G interrupt                       |
|   SRA    | 1         | Panel Self Refresh frame lock A interrupt |
|  LCDID   | 3         | LCD interrupt GPIO pin                    |
|    R     | 1         | Reserved, set to 0                        |

</div>

<div class="paragraph">

<div class="title">

Type

</div>

Descriptive name of each connector where only one signal may be
displayed through that connector at any given time.

</div>

<div class="paragraph">

If there is a breakout cable where multiple displays can be displayed
through that breakout cable at the same time, generally the type should
be listed as Breakout Cable Name - End connector, or Initial Connector -
Final Connector. Example: LFH - DVI-I - 1.

</div>

<div class="paragraph">

Because TV’s can allow more than one connector per TV encoder, all
connectors associated with the TV device must be grouped together. The
DCB Display Path will point to the first TV connector on the list. The
DCB TV DSI Connector Count field will list how many connectors are
available for the TV.

</div>

<div class="paragraph">

All devices are considered detachable (or removable) unless otherwise
noted.

</div>

<div class="paragraph">

Currently defined values are:

</div>

<div class="ulist">

  - 0x00 = VGA 15-pin connector

  - 0x01 = DVI-A

  - 0x02 = Pod - VGA 15-pin connector

  - 0x10 = TV - Composite Out

  - 0x11 = TV - S-Video Out

  - 0x12 = TV - S-Video Breakout - Composite (Used for board that list 2
    of the RGB bits in the TVDACs field)

  - 0x13 = TV - HDTV Component - YPrPb

  - 0x14 = TV - SCART Connector

  - 0x16 = TV - Composite SCART over the BLUE channel of EIAJ4120
    (D-connector)

  - 0x17 = TV - HDTV - EIAJ4120 Connector (aka D-connector)

  - 0x18 = Pod - HDTV - YPrPb

  - 0x19 = Pod - S-Video

  - 0x1A = Pod - Composite

  - 0x20 = DVI-I-TV-S-Video

  - 0x21 = DVI-I-TV-Composite

  - 0x22 = DVI-I-TV-S-Video Breakout-Composite (Used for board that list
    2 of the RGB bits in the TVDACs field)

  - 0x30 = DVI-I

  - 0x31 = DVI-D

  - 0x32 = Apple Display Connector (ADC)

  - 0x38 = LFH-DVI-I-1

  - 0x39 = LFH-DVI-I-2

  - 0x3C = BNC Connector

  - 0x40 = LVDS-SPWG-Attached (non-removeable)

  - 0x41 = LVDS-OEM-Attached (non-removeable)

  - 0x42 = LVDS-SPWG-Detached (removeable)

  - 0x43 = LVDS-OEM-Detached (removeable)

  - 0x45 = TMDS-OEM-Attached (non-removeable)

  - 0x46 = DisplayPort External Connector (as a special case, if the
    "Location" field is 0 and the "Platform" type in the Connector Table
    Header is 7 (Desktop with Integrated full DP), this indicates a
    non-eDP DisplayPort Internal Connector, which is non-removeable)

  - 0x47 = DisplayPort Internal Connector(non-removeable)

  - 0x48 = DisplayPort (Mini) External Connector

  - 0x50 = VGA 15-pin connector if not docked *'(See Notes below)*'

  - 0x51 = VGA 15-pin connector if docked *'(See Notes below)*'

  - 0x52 = DVI-I connector if not docked *'(See Notes below)*'

  - 0x53 = DVI-I connector if docked *'(See Notes below)*'

  - 0x54 = DVI-D connector if not docked *'(See Notes below)*'

  - 0x55 = DVI-D connector if docked *'(See Notes below)*'

  - 0x56 = DisplayPort External Connector if not docked *'(See Notes
    below)*'

  - 0x57 = DisplayPort External Connector if docked *'(See Notes
    below)*'

  - 0x58 = DisplayPort (Mini) External Connector if not docked *'(See
    Notes below)*'

  - 0x59 = DisplayPort (Mini) External Connector if docked *'(See Notes
    below)*'

  - 0x60 = 3-Pin DIN Stereo Connector

  - 0x61 = HDMI-A connector

  - 0x62 = Audio S/PDIF connector

  - 0x63 = HDMI-C (Mini) connector

  - 0x64 = LFH-DP-1

  - 0x65 = LFH-DP-2

  - 0x70 = Virtual connector for Wifi Display (WFD)

  - 0xFF = Skip Entry. This allows for quick removal of an entry from
    the Connector Table.

</div>

<div class="paragraph">

<div class="title">

Location

</div>

Specific locations depend on the platform type. The SW could define Real
location as ((Platform Type \<\< 4) | This Location Field) if it’s
easier to deal with a single number rather than two separate lists.
Generally, a value of 0 defines the South most connector, which is the
connector on the bracket closest to the AGP/PCI connector. The specific
values here are to be determined.

</div>

<div class="paragraph">

<div class="title">

Hotplug A

</div>

This field dictates if this connector triggers the Hotplug A interrupt.
If defined, then the Hotplug A interrupt must be defined inside the GPIO
Assignment table.

</div>

<div class="paragraph">

<div class="title">

Hotplug B

</div>

This field dictates if this connector triggers the Hotplug B interrupt.
If defined, then the Hotplug B interrupt must be defined inside the GPIO
Assignment table.

</div>

<div class="paragraph">

<div class="title">

DP2DVI A

</div>

This field indictates if this connector is connected to DP to DVI
present A. If defined, then the DisplayPort to DVI dongle A present must
be defined inside the GPIO Assignment table.

</div>

<div class="paragraph">

<div class="title">

DP2DVI B

</div>

This field indictates if this connector is connected to DP to DVI
present B. If defined, then the DisplayPort to DVI dongle B present must
be defined inside the GPIO Assignment table.

</div>

<div class="paragraph">

<div class="title">

Hotplug C

</div>

This field dictates if this connector triggers the Hotplug C interrupt.
If defined, then the Hotplug C interrupt must be defined inside the GPIO
Assignment table.

</div>

<div class="paragraph">

<div class="title">

Hotplug D

</div>

This field dictates if this connector triggers the Hotplug D interrupt.
If defined, then the Hotplug D interrupt must be defined inside the GPIO
Assignment table.

</div>

<div class="paragraph">

<div class="title">

DP2DVI C

</div>

This field indictates if this connector is connected to DP to DVI
present C. If defined, then the DisplayPort to DVI dongle C present must
be defined inside the GPIO Assignment table.

</div>

<div class="paragraph">

<div class="title">

DP2DVI D

</div>

This field indictates if this connector is connected to DP to DVI
present D. If defined, then the DisplayPort to DVI dongle D present must
be defined inside the GPIO Assignment table.

</div>

<div class="paragraph">

<div class="title">

DPAux/I2C Select A

</div>

This field indictates if this connector is connected to DPAUX/I2C select
A. If defined, then the DPAUX/I2C select A must be defined inside the
GPIO Assignment table.

</div>

<div class="paragraph">

<div class="title">

DPAux/I2C Select B

</div>

This field indictates if this connector is connected to DPAUX/I2C select
B. If defined, then the DPAUX/I2C select B must be defined inside the
GPIO Assignment table.

</div>

<div class="paragraph">

<div class="title">

DPAux/I2C Select C

</div>

This field indictates if this connector is connected to DPAUX/I2C select
C. If defined, then the DPAUX/I2C select C must be defined inside the
GPIO Assignment table.

</div>

<div class="paragraph">

<div class="title">

DPAux/I2C Select D

</div>

This field indictates if this connector is connected to DPAUX/I2C select
D. If defined, then the DPAUX/I2C select D must be defined inside the
GPIO Assignment table.

</div>

<div class="paragraph">

<div class="title">

Hotplug E

</div>

This field dictates if this connector triggers the Hotplug E interrupt.
If defined, then the Hotplug E interrupt must be defined inside the GPIO
Assignment table.

</div>

<div class="paragraph">

<div class="title">

Hotplug F

</div>

This field dictates if this connector triggers the Hotplug F interrupt.
If defined, then the Hotplug F interrupt must be defined inside the GPIO
Assignment table.

</div>

<div class="paragraph">

Hotplug G. This field dictates if this connector triggers the Hotplug G
interrupt. If defined, then the Hotplug G interrupt must be defined
inside the GPIO Assignment table.

</div>

<div class="paragraph">

<div class="title">

Panel Self Refresh Frame Lock A

</div>

This field dictates if this connector triggers the FrameLock A
interrupt.

</div>

<div class="paragraph">

<div class="title">

LCD ID

</div>

This field dictates if this connector is connected to LCD\# GPIO(s). If
defined, then the LCD\# GPIO(s) must be defined inside the GPIO
Assignment table. LCD ID field only applies to the connector types
listed below. All other types must set this field to 0. If defined, then
the FrameLock A interrupt must be defined inside the GPIO Assignment
table.

</div>

<div class="paragraph">

LCD ID only applies for these connector types:

</div>

<div class="ulist">

  - 0x40 = LVDS-SPWG-Attached (non-removeable)

  - 0x41 = LVDS-OEM-Attached (non-removeable)

  - 0x42 = LVDS-SPWG-Detached (removeable)

  - 0x43 = LVDS-OEM-Detached (removeable)

  - 0x45 = TMDS-OEM-Attached (non-removeable)

  - 0x46 = DisplayPort with Integrated Full DP (only if the special case
    described above in the *type* field’s entry 0x46 applies)

  - 0x47 = DisplayPort Internal Connector(non-removeable)

</div>

<div class="paragraph">

Special case for connector platform type 0x09 = MXM module, if DCB
connector type and MXM-SIS output connector type have below
combinations:

</div>

<div class="ulist">

  - 0x46 = DisplayPort External Connector and MXM-SIS connector type is
    0x07 = DISPLAYPORT\_INT

  - 0x46 = DisplayPort External Connector and MXM-SIS connector type is
    0x0E = EDP\_INT

</div>

<div class="paragraph">

Values are:

</div>

<div class="ulist">

  - 0 = Connected to LCD0 GPIO(s)

  - 1 = Connected to LCD1 GPIO(s)

  - 2 = Connected to LCD2 GPIO(s)

  - 3 = Connected to LCD3 GPIO(s)

  - 4 = Connected to LCD4 GPIO(s)

  - 5 = Connected to LCD5 GPIO(s)

  - 6 = Connected to LCD6 GPIO(s)

  - 7 = Connected to LCD7 GPIO(s)

</div>

<div class="sect3">

#### Connector Table Entry Notes

<div class="paragraph">

There are some connector types, 0x50 through 0x57, that require extra
code in the detection routines inside any code that uses the DCB. For
Mobile systems, some connectors might only be on the actual body of the
notebook. Also, some connectors might only show up on the docking
station. Therefore we need to make sure that we don’t allow anyone to
select a device that is not actually present. So, when we see connectors
with the "if not docked" and "if docked" text in the description, we
must make sure that our detection code checks the docked condition first
and possibly culls any further detection attempts if the docked
condition is not met.

</div>

</div>

<div class="sect1">

## HDTV Translation Table

<div class="sectionbody">

<div class="paragraph">

Two GPIO functions (HD Dongle Strap 0/1) are defined to allows users to
specify the format of an HDTV connected to the system via an external
switch inside a dongle. Only two pins have been assigned for more than 4
possible formats (9 as it now) because it is unlikely that a given GPU
board needs to support all formats. This array is indexed from those two
GPIOs which would define the HDTV format.

</div>

<div class="sect2">

### HDTV Translation Table Header

<div class="tableblock">

31

</div>

</div>

</div>

</div>

30

29

28

27

26

25

24

23

22

21

20

19

18

17

16

15

14

13

12

11

10

9

8

7

6

5

4

3

2

1

0

Entry Size = 1

Entry Count

Header Size = 4

Version = 0

<div class="paragraph">

<div class="title">

Version

</div>

Version \# of the HDTV Translation Table Header and Entries. The HDTV
Translation Table version will start with 0 in this field.

</div>

<div class="paragraph">

<div class="title">

Header Size

</div>

Size of the HDTV Translation Table in bytes. Initially, this is 4 bytes.

</div>

<div class="paragraph">

<div class="title">

Entry Count

</div>

Number of HDTV Translation Table Entries starting directly after the end
of this header.

</div>

<div class="paragraph">

<div class="title">

Entry Size

</div>

Size of each entry in bytes. Initially, this is 1 byte.

</div>

<div class="sect2">

### HDTV Translation Table Entry

<div class="tableblock">

7

</div>

</div>

6

5

4

3

2

1

0

Rsvd

HDStand

<div class="paragraph">

<div class="title">

HD Standard

</div>

This field lists the specific standard to use for this entry. Defined
values are:

</div>

<div class="ulist">

  - 0 = HD576i

  - 1 = HD480i

  - 2 = HD480p\_60

  - 3 = HD576p\_50

  - 4 = HD720p\_50

  - 5 = HD720p\_60

  - 6 = HD1080i\_50

  - 7 = HD1080i\_60

  - 8 = HD1080p\_24

</div>

<div class="sect1">

## Switched Outputs Table

<div class="sectionbody">

<div class="paragraph">

There are new designs that allow to change the routing of device
detection, selection, switching and I2C switching by way of a GPIO. This
table assigns the relationship of the routing to specific DCB indices.

</div>

<div class="sect2">

### Switched Outputs Table Header

<div class="tableblock">

31

</div>

</div>

</div>

</div>

30

29

28

27

26

25

24

23

22

21

20

19

18

17

16

15

14

13

12

11

10

9

8

7

6

5

4

3

2

1

0

Entry Size = 5

Entry Count

Header Size = 4

Version = 0x10

<div class="paragraph">

<div class="title">

Version

</div>

Version \# of the Switched Outputs Table Header and Entries. The
Switched Outputs Table will version will start with 0x10 in this field.

</div>

<div class="paragraph">

<div class="title">

Header Size

</div>

Size of the Switched Outputs Table in bytes.

</div>

<div class="paragraph">

<div class="title">

Entry Count

</div>

Number of Switched Outputs Table Entries starting directly after the end
of this header.

</div>

<div class="paragraph">

<div class="title">

Entry Size

</div>

Size of each entry in bytes. Initially, this is 5 bytes.

</div>

<div class="sect2">

### Switched Outputs Table Entry

<div class="tableblock">

39

</div>

</div>

38

37

36

35

34

33

32

31

30

29

28

27

26

25

24

23

22

21

20

19

18

17

16

15

14

13

12

11

10

9

8

7

6

5

4

3

2

1

0

R

S

GPIO\#

T

R

S

GPIO\#

T

R

S

GPIO\#

T

R

S

GPIO\#

T

Rsvd

DCBx

<div class="tableblock">

|  Name  | Bit width | Values and Meaning                       |
| :----: | :-------- | :--------------------------------------- |
|  DCBx  | 5         | DCB table index for this entry           |
|  Rsvd  | 3         | Reserved, set to 0                       |
|   T    | 1         | Device Selection GPIO Type, 1 = external |
| GPIO\# | 5         | Device Selection GPIO Number             |
|   S    | 1         | Device Selection GPIO State              |
|   R    | 1         | Reserved, set to 0                       |

</div>

<div class="paragraph">

<div class="title">

DCB Index

</div>

This index is used to determine which entry in the DCB table this
Switched Output Table entry goes with.

</div>

<div class="paragraph">

<div class="title">

Device Selection GPIO Type

</div>

This flag determines the location of the control for the GPIO that
controls Device Selection. Defined values are:

</div>

<div class="ulist">

  - 0 = Internal GPIO or GPU controlled GPIO

  - 1 = External GPIO

</div>

<div class="paragraph">

<div class="title">

Device Selection GPIO Number

</div>

This field describes the GPIO number that controls device selection. If
the value is set to 0x1F, then this functionality is not used. The

</div>

<div class="paragraph">

<div class="title">

Device Selection GPIO State

</div>

This flag tells the logical GPIO state in order to select or enable the
associated DCB index for this entry. The physical logic here is found in
the [GPIO Assignment Table](#_gpio_assignment_table). Defined values
are:

</div>

<div class="ulist">

  - 0 = Logical OFF state.

  - 1 = Logical ON state.

</div>

<div class="paragraph">

<div class="title">

Device Detection Switching GPIO Type

</div>

This flag determines the location of the control for the GPIO that
controls Device Detection. Defined values are:

</div>

<div class="ulist">

  - 0 = Internal GPIO or GPU controlled GPIO

  - 1 = External GPIO

</div>

<div class="paragraph">

<div class="title">

Device Detection Switching GPIO Number

</div>

This field describes the GPIO number that controls Device Detection. If
the value is set to 0x1F, then this functionality is not used. In order
to run detection, this GPIO must be set to the Device Detection
Switching GPIO State before reading the state.

</div>

<div class="paragraph">

<div class="title">

Device Detection Switching GPIO State

</div>

This flag tells the logical GPIO state in order to detect the associated
DCB index for this entry. The physical logic here is found in the [GPIO
Assignment Table](#_gpio_assignment_table). Defined values are:

</div>

<div class="ulist">

  - 0 = Logical OFF state.

  - 1 = Logical ON state.

</div>

<div class="paragraph">

<div class="title">

Device Detection Load GPIO Type

</div>

This flag determines the location of the input for the GPIO that returns
the Device Detection Load. Defined Values are:

</div>

<div class="ulist">

  - 0 = Internal GPIO or GPU controlled GPIO

  - 1 = External GPIO

</div>

<div class="paragraph">

<div class="title">

Device Detection Load GPIO Number

</div>

This field describes the GPIO number that returns Device Detection Load.
If the value is set to 0x1F, then this functionality is not used. When
running detection, the Devices Detecion Switching GPIO must be set to
the Device Detection Switching GPIO State. Then read this GPIO to get
the Load State.

</div>

<div class="paragraph">

<div class="title">

Device Detection Load GPIO State

</div>

This flag tells the physical GPIO state that indicates a connected
state. Defined values are:

</div>

<div class="ulist">

  - 0 = If the GPIO reads back physically as 0, then the device is
    connected.

  - 1 = If the GPIO reads back physically as 1, then the device is
    connected.

</div>

<div class="paragraph">

<div class="title">

DDC Port Switching GPIO Type

</div>

This flag determines the location of the control for the GPIO that
controls the DDC port. Defined Values are:

</div>

<div class="ulist">

  - 0 = Internal GPIO or GPU controlled GPIO

  - 1 = External GPIO

</div>

<div class="paragraph">

<div class="title">

DDC Port Switching GPIO Number

</div>

This field describes the GPIO number that controls the routing of the
DDC Port. If the value is set to 0x1F, then this functionality is not
used. In order to use the DDC Port, this GPIO must be set to the DDC
Port Switching GPIO State.

</div>

<div class="paragraph">

<div class="title">

DDC Port Switching GPIO State

</div>

This flag tells the logical GPIO state in order to use the DDC Port for
this DCB index entry. The physical logic here is found in the [GPIO
Assignment Table](#_gpio_assignment_table). Defined values are:

</div>

<div class="ulist">

  - 0 = Logical OFF state.

  - 1 = Logical ON state.

</div>

<div id="footnotes">

-----

</div>

<div id="footer">

<div id="footer-text">

Last updated 2014-12-08

</div>

</div>
