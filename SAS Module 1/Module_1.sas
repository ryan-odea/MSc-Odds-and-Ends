data work.shoes;
	set sashelp.shoes;
	NetSales=Sales-Returns;
run;

proc means data=work.shoes mean sum; 
	var NetSales;
	class region;
run;