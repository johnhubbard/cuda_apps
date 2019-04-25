<div id="header">

# NVIDIA Memory Tweak Table Specifications

</div>

<div id="preamble">

<div class="sectionbody">

</div>

</div>

## Purpose

<div class="sectionbody">

<div class="paragraph">

This document describes the VBIOS Memory tweak table entries.

</div>

### Memory Tweak Table Header

<div style="clear:left">

</div>

<div class="tableblock">

<table style="width:99%;">
<colgroup>
<col style="width: 22%" />
<col style="width: 11%" />
<col style="width: 66%" />
</colgroup>
<thead>
<tr class="header">
<th style="text-align: left;">FieldName</th>
<th style="text-align: center;">Size (in bits)</th>
<th style="text-align: left;">Description</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;"><div class="verse">
Version
</div></td>
<td style="text-align: center;"><p>8</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> Memory Tweak Table Version (0x20) </dt>
<dd>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><div class="verse">
Header Size
</div></td>
<td style="text-align: center;"><p>8</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> Size of Memory Tweak Table Header in bytes (6) </dt>
<dd>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><div class="verse">
Base Entry Size
</div></td>
<td style="text-align: center;"><p>8</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> Size of Memory Tweak Table Base Entry in bytes (76) </dt>
<dd>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><div class="verse">
Extended Entry Size
</div></td>
<td style="text-align: center;"><p>8</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> Size of Memory Tweak Table Extended Entry in bytes (12) </dt>
<dd>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><div class="verse">
Extended Entry Count
</div></td>
<td style="text-align: center;"><p>8</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> Number of Memory Tweak Table Extended Entries per Memory Tweak Table Entry </dt>
<dd>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><div class="verse">
Entry Count
</div></td>
<td style="text-align: center;"><p>8</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> Number of Memory Tweak Table Entries (combined Base Entry plus Extended Entry Count of Extended Entries) </dt>
<dd>
</dd>
</dl>
</div>
</div></td>
</tr>
</tbody>
</table>

</div>

### Memory Tweak Table Entry

<div style="clear:left">

</div>

<div class="paragraph">

Each entry is made up of a single Base Entry and multiple Extended
Entries. The entire size of an entry is given by (
MemoryTweakTableHeader.BaseEntrySize +
MemoryTweakTableHeader.ExtendedEntrySize ×
MemoryTweakTableHeader.ExtendedEntryCount ).

</div>

### Memory Tweak Table Base Entry

<div style="clear:left">

</div>

<div class="tableblock">

<table style="width:98%;">
<colgroup>
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 66%" />
</colgroup>
<thead>
<tr class="header">
<th style="text-align: left;">FieldName</th>
<th style="text-align: center;">Size (in bits)</th>
<th style="text-align: left;">Description</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;"><p>CONFIG0</p></td>
<td style="text-align: center;"><p>32</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> Field Definitions </dt>
<dd><p>[7:0] = RC<br />
[16:8] = RFC<br />
[23:17]= RAS<br />
[30:24]= RP<br />
[31:31]= Reserved</p>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>CONFIG1</p></td>
<td style="text-align: center;"><p>32</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> Field Definitions </dt>
<dd><p>[6:0] = CL<br />
[13:7] = WL<br />
[19:14]= RD_RCD<br />
[25:20]= WR_RCD<br />
[31:26]= Reserved</p>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>CONFIG2</p></td>
<td style="text-align: center;"><p>32</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> Field Definitions </dt>
<dd><p>[3:0] = RPRE<br />
[7:4] = WPRE<br />
[14:8] = CDLR<br />
[22:16] = WR<br />
[27:24] = W2R_BUS<br />
[31:28] = R2W_BUS</p>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>CONFIG3</p></td>
<td style="text-align: center;"><p>32</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> Field Definitions </dt>
<dd><p>[4:0] = PDEX<br />
[8:5] = PDEN2PDEX<br />
[16:9] = FAW<br />
[23:17] = AOND<br />
[27:24] = CCDL<br />
[31:28] = CCDS</p>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>CONFIG4</p></td>
<td style="text-align: center;"><p>32</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> Field Definitions </dt>
<dd><p>[2:0] = REFRESH_LO<br />
[14:3] = REFRESH<br />
[20:15] = RRD<br />
[26:21] = DELAY0<br />
[31:27] = Reserved<br />
</p>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>CONFIG5</p></td>
<td style="text-align: center;"><p>32</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> Field Definitions </dt>
<dd><p>[2:0] = ADR_MIN<br />
[3:3] = Reserved<br />
[10:4] = WRCRC<br />
[11:11] = Reserved<br />
[17:12] = OFFSET0<br />
[19:18] = DELAY0_MSB<br />
[23:20] = OFFSET1<br />
[27:24] = OFFSET2<br />
[31:28] = DELAY0</p>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Reserved</p></td>
<td style="text-align: center;"><p>184</p></td>
<td style="text-align: left;"><div>

</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Drive Strength</p></td>
<td style="text-align: center;"><p>2</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> Drive strength value to program depending on memory type </dt>
<dd><p>SDDR2: MR1[1:1] - Output Driver Impedence Control<br />
SDDR3: Unused<br />
GDDR3: MR1[1:0] = Driver Strength<br />
GDDR5: MR1[1:0] = Driver Strength</p>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Voltage0</p></td>
<td style="text-align: center;"><p>3</p></td>
<td style="text-align: left;"><div>

</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Voltage1</p></td>
<td style="text-align: center;"><p>3</p></td>
<td style="text-align: left;"><div>

</div></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Voltage2</p></td>
<td style="text-align: center;"><p>3</p></td>
<td style="text-align: left;"><div>

</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>R2P</p></td>
<td style="text-align: center;"><p>5</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> Minimum number of cycles from a read command to a precharge command for the same bank. </dt>
<dd>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Voltage3</p></td>
<td style="text-align: center;"><p>3</p></td>
<td style="text-align: left;"><div>

</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Reserved</p></td>
<td style="text-align: center;"><p>1</p></td>
<td style="text-align: left;"><div>

</div></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Voltage4</p></td>
<td style="text-align: center;"><p>3</p></td>
<td style="text-align: left;"><div>

</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Reserved</p></td>
<td style="text-align: center;"><p>1</p></td>
<td style="text-align: left;"><div>

</div></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Voltage5</p></td>
<td style="text-align: center;"><p>3</p></td>
<td style="text-align: left;"><div>

</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Reserved</p></td>
<td style="text-align: center;"><p>5</p></td>
<td style="text-align: left;"><div>

</div></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>RDCRC</p></td>
<td style="text-align: center;"><p>4</p></td>
<td style="text-align: left;"><div>

</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Reserved</p></td>
<td style="text-align: center;"><p>36</p></td>
<td style="text-align: left;"><div>

</div></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>TIMING22</p></td>
<td style="text-align: center;"><p>32</p></td>
<td style="text-align: left;"><div>
<div class="literalblock">
<div class="content">
<pre><code>Field Definitions</code></pre>
</div>
</div>
<div class="literalblock">
<div class="content">
<pre><code>[9:0]   = RFCSBA</code></pre>
</div>
</div>
<div class="literalblock">
<div class="content">
<pre><code>[17:10] = RFCSBR</code></pre>
</div>
</div>
<div class="literalblock">
<div class="content">
<pre><code>[31:18] = Reserved</code></pre>
</div>
</div>
</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Reserved</p></td>
<td style="text-align: center;"><p>128</p></td>
<td style="text-align: left;"><div>

</div></td>
</tr>
</tbody>
</table>

</div>

### Memory Tweak Table Extended Entry

<div style="clear:left">

</div>

<div class="tableblock">

| FieldName | Size (in bits) | Description |
| :-------- | :------------: | :---------- |
| Reserved  |       96       |             |

</div>

</div>

<div id="footer">

<div id="footer-text">

Last updated 2018-01-26 11:45:36 PDT

</div>

</div>
