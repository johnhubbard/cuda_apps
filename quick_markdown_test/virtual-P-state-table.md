<div id="header">

# NVIDIA Virtual P-state Table Specification

</div>

<div id="content">

<div class="sect1">

## Purpose

<div class="sectionbody">

<div class="paragraph">

This document describes the Virtual P-state (vP-state) Table in the
NVIDIA VBIOS.

</div>

<div class="paragraph">

The Virtual P-state (vP-state) Table maps discreet points on the
Voltage/Frequency curve to vP-states. These vP-states are typically used
as caps (limits).

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
<td>This specification only provides the details about vP-state entry which is required to fetch base clock.</td>
</tr>
</tbody>
</table>

</div>

</div>

</div>

<div class="sect1">

## Virtual P-state Table

<div class="sectionbody">

<div class="paragraph">

The Virtual P-state Table starts with a header, followed immediately by
an array of entries.

</div>

<div class="paragraph">

It consist of following sections:

</div>

<div class="ulist">

  - Header – The version number, header size, size of each vP-state
    entry, number of vP-state entries, etc.

  - Entry – One for each vP-state. It consist of associated P-state.

  - Domain frequencies – Sub-table in vP-state entry table consisting of
    limits for domain frequencies.

</div>

<div class="paragraph">

The vP-state table is a part of BIOS information table’s (BIT)
Performance table entry (ID = ‘P’ i.e. 0x50 and version = 2) and its
location is extracted by reading 32 bit dword from an offset 0x38 from
the performance table (refer to [sample code](#_sample_code)).

</div>

<div class="sect2">

### Virtual P-state Table Version

<div class="paragraph">

This document describes vP-state table version 1.0 which is supported
only on GPU families GF11X through GM20X (i.e. NVC0 through NV110
families as per nouveau’s code names).

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
<td>Table version and structure will change in Pascal and later GPUs.</td>
</tr>
</tbody>
</table>

</div>

</div>

<div class="sect2">

### Virtual P-state Table Header Structure

<div class="tableblock">

| Name                                     | Bit width | Meaning and (Values)                                                                                                           |
| :--------------------------------------- | --------: | :----------------------------------------------------------------------------------------------------------------------------- |
| Version                                  |         8 | Virtual P-state Table Version (0x10)                                                                                           |
| Header Size                              |         8 | Size of Virtual P-state Table Header in bytes (20)                                                                             |
| Base Entry Size                          |         8 | Size of Virtual P-state Table Entry in bytes, not including the domain frequencies (5)                                         |
| Domain Freq Size                         |         8 | Size of each Virtual P-state Domain Frequency Entry in bytes (2)                                                               |
| Domain Freq Count                        |         8 | Number of Virtual P-state Domain Frequencies allocated                                                                         |
| Entry Count                              |         8 | Number of Virtual P-state Table Entries                                                                                        |
| Reserved                                 |        88 |                                                                                                                                |
| Index of Rated TDP vP-state (Base Clock) |         8 | Fastest thermally sustainable vP-state for the TDP app on worst case silicon, worst case conditions (also known as base clock) |
| Reserved                                 |        40 |                                                                                                                                |

</div>

</div>

<div class="sect2">

### Virtual P-state Table Entry Structure

<div class="tableblock">

| Name     | Bit width | Values and Meaning                                                           |
| :------- | --------: | :--------------------------------------------------------------------------- |
| P-state  |         8 | P-state associated with this vP-state. A value of 0xff indicates SKIP ENTRY. |
| Reserved |        32 |                                                                              |

</div>

</div>

<div class="sect2">

### Virtual P-state Domain Frequency Entry Structure

<div class="paragraph">

This is a sub-table in Virtual P-state Table Entry. Domain frequency
entries are indexed as per clock domain enumeration.

</div>

<div class="tableblock">

| Name             | Bit width | Values and Meaning                                                                                                                   |
| :--------------- | --------: | :----------------------------------------------------------------------------------------------------------------------------------- |
| Domain Frequency |        16 | Domain frequency associated with this vP-state in MHz. A value of 0 indicates that vP-states do not specify a limit for this domain. |

</div>

<div class="paragraph">

It is safe to assume that there will always be one domain frequency
entry per vP-state table entry. Non-zero value of this entry belongs to
GPC clock domain.

</div>

</div>

</div>

</div>

<div class="sect1">

## Sample code

<div class="sectionbody">

<div class="paragraph">

Sample code to read base clock from vP-state table.

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
<td>this code is based on nouveau driver which is capable of reading various entries from BIT.</td>
</tr>
</tbody>
</table>

</div>

<div class="listingblock">

<div class="content">

``` 
    1: void read_vpstate_table(struct nvkm_bios *bios)
    2: {
    3:         struct bit_entry bit_P;
    4:         u16 vpstate_tbl = 0x0000, offset, domain_clk;
    5:         u8 ver, i, hdr_size, base_entry_size, domain_freq_size, domain_freq_count, base_clk_idx;
    6:
    7:         if (!bit_entry(bios, 'P', &bit_P)) {
    8:                 if (bit_P.version == 2)
    9:                         vpstate_tbl = nvbios_rd16(bios, bit_P.offset + 0x38);
   10:                 else
   11:                         return;
   12:
   13:                 if (vpstate_tbl) {
   14:                         ver = nvbios_rd08(bios, vpstate_tbl + 0);
   15:                         printk("vP-state entry version = 0x%x\n", ver);
   16:                         switch (ver) {
   17:                         case 0x10:
   18:                                 printk("vP-state header size = %d\n",
   19:                                         hdr_size = nvbios_rd08(bios, vpstate_tbl + 1));
   20:                                 printk("base entry size = %d\n",
   21:                                         base_entry_size = nvbios_rd08(bios, vpstate_tbl + 2));
   22:                                 printk("domain freq entry size = %d\n",
   23:                                         domain_freq_size = nvbios_rd08(bios, vpstate_tbl + 3));
   24:                                 printk("domain freq entries count = %d\n",
   25:                                         domain_freq_count = nvbios_rd08(bios, vpstate_tbl + 4));
   26:
   27:
   28:                                 base_clk_idx = nvbios_rd08(bios, vpstate_tbl + 17);
   29:                                 printk("base clk index = %d\n", base_clk_idx);
   30:
   31:
   32:                                 offset = vpstate + hdr_size +
   33:                                         ((base_entry_size + (domain_freq_size * domain_freq_count)) * base_clk);
   34:                                 printk("offset = %d\n", offset);
   35:
   36:
   37:                                 printk("p-state for base clk = %d",
   38:                                         nvbios_rd08(bios, offset + 0));
   39:
   40:                                 for (i = 0; i < domain_freq_count; i++) {
   41:                                         domain_clk = nvbios_rd16(bios, offset + base_entry_size +
   42:                                                                  (i * domain_freq_size));
   43:                                         printk("domain clock index = %d, domain clock limit value = %d MHz\n",
   44:                                                 i, domain_clk);
   45:                                 }
   46:
   47:                                 return;
   48:                         default:
   49:                                 return;
   50:                         }
   51:                 }
   52:         }
   53:
   54:         return;
   55: }
```

</div>

</div>

</div>

</div>

</div>

<div id="footnotes">

-----

</div>

<div id="footer">

<div id="footer-text">

Last updated 2016-03-28

</div>

</div>
