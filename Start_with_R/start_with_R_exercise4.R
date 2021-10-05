read_csv("../W4_start_with_R/SAFI_clean.csv")
library(tidyverse)
library(here)
interviews <- read_csv(
  here("SAFI_clean.csv"), 
  na = "NULL")
