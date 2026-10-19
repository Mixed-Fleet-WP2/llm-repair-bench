A[] (carrier.Transport imply bin_fill <= BIN_CAPACITY)
A[] (carrier.LoadBin imply carrier.t <= 4)
E<> sorter.Deposit
E<> carrier.Done
