<div id="header">

# NVIDIA Memory Clock Table Specifications

</div>

<div id="preamble">

<div class="sectionbody">

</div>

</div>

## Purpose

<div class="sectionbody">

<div class="paragraph">

This document describes the VBIOS Memory clock table entries. The Memory
Clock Table starts with a header, followed immediately by an array of
entries.

</div>

### Memory Clock Table Header

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
<td style="text-align: left;"><p>Version</p></td>
<td style="text-align: center;"><p>8</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> Memory Clock Table Version (0x11) </dt>
<dd>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Header Size</p></td>
<td style="text-align: center;"><p>8</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> Size of Memory Clock Table Header in bytes (26) </dt>
<dd>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Base Entry Size</p></td>
<td style="text-align: center;"><p>8</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> Size of Memory Clock Table Base Entry in bytes (20) </dt>
<dd>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Strap Entry Size</p></td>
<td style="text-align: center;"><p>8</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> Size of Memory Clock Table Strap Entry in bytes (26) </dt>
<dd>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Strap Entry Count</p></td>
<td style="text-align: center;"><p>8</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> Number of Memory Clock Table Strap Entries per Memory Clock Table Entry </dt>
<dd>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Entry Count</p></td>
<td style="text-align: center;"><p>8</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> Number of Memory Clock Table Entries (combined Base Entry plus Strap Entry Count of Strap Entries) </dt>
<dd>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Reserved</p></td>
<td style="text-align: center;"><p>160</p></td>
<td style="text-align: left;"><div>

</div></td>
</tr>
</tbody>
</table>

</div>

### Memory Clock Table Base Entry

<div style="clear:left">

</div>

<div class="paragraph">

Each entry is made up of a single Base Entry and multiple Strap Entries.
The entire size of an entry is given by (
MemoryClockTableHeader.BaseEntrySize +
MemoryClockTableHeader.StrapEntrySize ×
MemoryClockTableHeader.StrapEntryCount ). Each entry provides
information needed for operating the memory at a frequency between
MemoryClockTableBaseEntry.Minimum.Frequency and
MemoryClockTableBaseEntry.Maximum.Frequency, inclusively.

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
<td style="text-align: left;"><p>Min Frequency</p></td>
<td style="text-align: center;"><p>16</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> [15:14] = Reserved<br />
[13:0] = Frequency (MHz) </dt>
<dd>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Max Frequency</p></td>
<td style="text-align: center;"><p>16</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> [15:14] = Reserved<br />
[13:0] = Frequency(MHz) </dt>
<dd>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Reserved</p></td>
<td style="text-align: center;"><p>40</p></td>
<td style="text-align: left;"><div>

</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Read/Write Config0</p></td>
<td style="text-align: center;"><p>32</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> [8:0] = Read Setting0<br />
[17:9] = Write Settings0<br />
[19:18] = Reserved<br />
[24:20] = ReadSettings1<br />
[31:25] = Reserved </dt>
<dd>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Read/Write Config1</p></td>
<td style="text-align: center;"><p>32</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> [3:0] = Read Settings0<br />
[7:4] = Write Settings0<br />
[11:8] = Read Settings1<br />
[15:12] = Write Settings1<br />
[19:16] = Read Settings2<br />
[23:20] = Write Settings2<br />
[31:24] = Timing Settings0 </dt>
<dd>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Reserved</p></td>
<td style="text-align: center;"><p>24</p></td>
<td style="text-align: left;"><div>

</div></td>
</tr>
</tbody>
</table>

</div>

### Memory Clock Table Strap Entry

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
<td style="text-align: left;"><p>MemTweak Index</p></td>
<td style="text-align: center;"><p>8</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> [7:0] MemTweak Index </dt>
<dd>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Flags0</p></td>
<td style="text-align: center;"><p>8</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> [6:0] = Reserved<br />
[7:7] = Alignment Mode </dt>
<dd><p>0x0 = Phase detector (Default)<br />
0x1 = Pin</p>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Reserved</p></td>
<td style="text-align: center;"><p>48</p></td>
<td style="text-align: left;"><div>

</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Flags4</p></td>
<td style="text-align: center;"><p>8</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> [6:0] = Reserved<br />
[7:7] = MRS7 GDDR5 </dt>
<dd><p>0x0 = Disable (Default)<br />
0x1 = Enable</p>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Reserved</p></td>
<td style="text-align: center;"><p>8</p></td>
<td style="text-align: left;"><div>

</div></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p>Flags5</p></td>
<td style="text-align: center;"><p>8</p></td>
<td style="text-align: left;"><div>
<div class="dlist">
<dl>
<dt> [5:0] = Reserved<br />
[6:6] = GDDR5x Internal VrefC </dt>
<dd><p>0x0 = Disable (Default) (70% VrefC)<br />
0x1 = Enable (50% VrefC)</p>
</dd>
<dt> [7:7] = Reserved </dt>
<dd>
</dd>
</dl>
</div>
</div></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p>Reserved</p></td>
<td style="text-align: center;"><p>120</p></td>
<td style="text-align: left;"><div>

</div></td>
</tr>
</tbody>
</table>

</div>

</div>

<div id="footer">

<div id="footer-text">

Last updated 2018-01-26 11:44:35 PDT

</div>

</div>
