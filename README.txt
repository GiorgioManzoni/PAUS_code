002: it contains all the queries to the different tables separately.

003: just to check if the production table from COSMOHUB is readble or not.

004: Correct query (suggested by Jorge) for combining PAUS and COSMOS data. Also trying to study the number of filters.

005: clear queries to isolate cosmos from paus but most important the definition of the survey class.

006: just trying to download the deprecated table but it is too big.

007: improvement of the class survey define in 005 (I can now plot data with the luminosity cut that I want), without using interactive queries but just loading from file. 
     The files that I use are just the total cosmos sample and the paus matched with the cosmos information (jorge query).
     Remember that I am using the production 701 (that is a photoz production) and in this production all the observed paus 
     data belong to the cosmos field (i.e. I am not loosing PAUS data when doing the join with COSMOS).  

008 - 011: comparison of the half-light radius in the following catalogues: COSMOS / CFHTLS-D2 / CFHTLS-D3 / CFHTLenS-W3 in order to understand if CFHTLenS-W3 is a good parent catalogue for PAUS-W3 (i.e. if it has the same definition of half-light radius as in COSMOS which has already been used for reducing the PAUS data in the COSMOS field).

011: it contains both classes and plots that use those classes and it creates all the useful files that I used as an input to topcat to match objects in overlapping fields. It's a cleaner version of 010 where I avoided to plot things that are not useful.
——> !!! —> IN THE LAST PART OF THE CODE YOU FIND HOW TO QUERY VIZIER WITH PTYHON (…and successfuly get CFHTLS_W3) 
————> It also contains the function median_in_bins_err() which compute the median and the desired percentiles as an error bar binning the x-axis.
—> Analysis of the i'/y' filter (using Ifl that is the flag which identify the filter).

012: it contains all the classes that I created to manage different surveys. Ideally it would be great to use it as a library and include it in new jupiter files (but first I need to learn how to include a jupyter in another jupyter)

013: Trying to import the classes for all the surveys that I am using directly from a jupyter notebook but it doesn't work for now.

014: Copied by 012. I want to make a comparison between CFHTLS-W3 and CFHTLenS-W3 but I created this new file to keep clear 012 with just the classes and no rubbish. In 014 you can find:
—> The ra/dec plots of W3 with overlapping data from CFHTLS/W3 and CFHTLenS/W3. It can be seen that the masking is very different with Lens escluding lots of galaxies in big holes also around faint stars. Different zooms of the plot and also with a cut at i<23.
—> example of how to merge an additional column to a table in pandas (in this case I added scalelength to the lens table).
—> I matched CFHTLS/W3 with CFHTLenS/W3 using topcat (remember to use the option 'topcat -Xmx4096M' to allocate more memory since we are dealing with huge tables). In 014 I just check that the table is readable.
—> Analysis of FLUX_RADIUS against scalelength (with the good plot that I showed during the meeting in London). It seems that scalelength is the deconvolved version of FLUX_RADIUS but the analytic relation must be found (probably in Hildebrandt+2012).

015: Meant to be used for reading the new photo-z production 852 from paudm.

016: To be used as a template to merge a new column to an existing table using pandas.

017: The idea is to use the classes from 012 to compare CFHTLS/W3 masking with CFHTLenS/W3 ones. The idea comes from the plot in 014 showing the big holes in LenS that are not present in CFHTLS. Hendrik says that this holes appear just if we select MASK=0 so I want to explore other values of MASK.