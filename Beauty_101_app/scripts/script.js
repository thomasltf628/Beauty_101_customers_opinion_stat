$(document).ready(function(){
	 $("p.abs").hover(function(){
    		$(this).css("background-color", "#ffddf4");
	},
	 function(){
  		$(this).css("background-color", "");
	});
	 $("p.new").mouseleave(function(){
    		alert("You've reached the end of the page! Please come back!");
  	});
});

