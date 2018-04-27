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

012: it contains all the classes that I created to manage different surveys. Ideally it would be great to use it as a library and include it in new jupiter files (but first I need to learn how to include a jupyter in another jupyter)