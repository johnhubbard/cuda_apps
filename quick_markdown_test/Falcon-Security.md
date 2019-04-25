<div id="header">

# NVIDIA Falcon Security

</div>

<div id="content">

<div id="preamble">

<div class="sectionbody">

<div class="paragraph">

NVIDIA GPUs embed several microprocessors based on a custom architecture
called "Falcon". Starting with the Maxwell family of GPUs, these
microprocessors are changing to be able to better protect the hardware
from being misprogrammed.

</div>

</div>

</div>

<div class="sect1">

## Falcon security modes

<div class="sectionbody">

<div class="paragraph">

A Falcon microprocessor supporting advanced security modes can run in
one of three modes. Not all Falcon microprocessors on a GPU support all
modes.

</div>

<div class="ulist">

  - Non-secure (NS). In this mode, functionality is similar to Falcon
    architectures before security modes were introduced (pre-Maxwell),
    but capability is restricted. In particular, certain registers may
    be inaccessible for reads and/or writes, and physical memory access
    may be disabled (on certain Falcon instances). This is the only
    possible mode that can be used if you don’t have microcode
    cryptographically signed by NVIDIA.

  - Heavy Secure (HS). In this mode, the microprocessor is a black
    box — it’s not possible to read or write any Falcon internal
    state or Falcon registers from outside the Falcon (for example, from
    the host system). The only way to enable this mode is by loading
    microcode that has been signed by NVIDIA. (The loading process
    involves tagging the IMEM block as secure, writing the signature
    into a Falcon register, and starting execution. The hardware will
    validate the signature, and if valid, grant HS privileges.)

  - Light Secure (LS). In this mode, the microprocessor has more
    privileges than NS but fewer than HS. Some of the microprocessor
    state is visible to host software to ease debugging. The only way to
    enable this mode is by HS microcode enabling LS mode. Some
    privileges available to HS mode are not available here. LS mode is
    introduced in GM20x.

</div>

</div>

</div>

<div class="sect1">

## GM10x

<div class="sectionbody">

<div class="paragraph">

The intent for GM10x is to protect fuses and ROM from being written by
incorrect or malicious software.

</div>

<div class="paragraph">

This is implemented by preventing access to select GPU registers from
anything other than a Falcon running in a secure mode.

</div>

</div>

</div>

<div class="sect1">

## GM20x

<div class="sectionbody">

<div class="paragraph">

The intent for GM20x is to improve upon the GM10x implementation and add
some protection to the configuration of the hardware thermal shutdown
mechanism.

</div>

<div class="paragraph">

In addition to the registers protected by GM10x:

</div>

<div class="ulist">

  - Thermal shutdown registers are protected and can only be written
    from a secure microprocessor context. These registers can be broken
    down into two categories:
    
    <div class="ulist">
    
      - Thermal sensor setup
    
      - The temperature beyond which hardware triggers a forced shutdown
        to prevent damage.
    
    </div>

  - I<sup>2</sup>C bus C writes are restricted to a secure context, to
    prevent misprogramming thermal sensors.

  - A new mechanism is introduced to prevent microcode tampering after
    load. This is achieved by placing microcode in a write-protected
    region of memory.

  - Physical memory access restrictions are introduced. On all Falcons
    other than PMU (the "kitchen sink" Falcon) and DPU (the Falcon that
    services display), microprocessors running in NS mode will be unable
    to access physical memory (they may use virtual memory exclusively).
    In particular, this includes all microprocessors which perform work
    directly in response to userspace requests.

  - Devinit scripts are signed and executed on the PMU so that these
    scripts can configure protected registers like thermal shutdown
    parameters.

</div>

</div>

</div>

</div>

<div id="footnotes">

-----

</div>

<div id="footer">

<div id="footer-text">

Last updated Tue Sep 23 08:31:37 PDT 2014

</div>

</div>
