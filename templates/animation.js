var canvas = document.getElementById("mCanvas");
var context = canvas.getContext("2d");
var count = 0;
var arrPoints = [-1, -0.5, 0.5, 1];
var frameCount = 10;
var frameRate = 20;

//To get the reference of RequestAnimationFrame
window.reqAnimFrame = (function(callback) {
  return window.requestAnimationFrame || window.webkitRequestAnimationFrame || window.mozRequestAnimationFrame ||
    window.oRequestAnimationFrame || window.msRequestAnimationFrame;
})();

//Assigning Width and Height of Canvas
canvas.width = 300;
canvas.height = 300;
canvas.style = "position: absolute; top: 0px; left: 1250px; right: 0px; bottom: 830px; margin: auto";



// animateDog - Method to animate dog movement
//
var Dog = {
		imgDog: new Image(),
		frameHeight: 0,
		frameIndex: 0,
		xDog: 2 * canvas.width / 5,
		yDog:165,
		animateDog: function() {
			  var currFrameX;
			  if (this.frameIndex > 0 && this.frameIndex % 5 === 0) {
				  this.frameIndex = 0;
				  this.frameHeight += 61;
			  }
			  if (count === 9) {
				  this.frameIndex = 0;
				  this.frameHeight = 0;
				  count = 0;
			  }
			  currFrameX = 82 * (this.frameIndex % frameCount);
			  this.imgDog.src = 'pug-running_transparent.png';
			  context.drawImage(this.imgDog, currFrameX, this.frameHeight, 82, 61, this.xDog, this.yDog, 82, 61);
			} 
}

// drawRect - To create Window size background and surface rectangle. It creates two separate paths on the canvas

function drawRect() {

  var grd = context.createLinearGradient(0, 0, canvas.width, canvas.height);
  context.fillStyle = 'rgba(225,225,225,0.5)';
  context.fillRect(25,72,32,32);

  context.beginPath();
  context.strokeStyle = "white";
  context.moveTo(canvas.width / 4, 3 * canvas.height / 4);
  context.lineTo(3 * canvas.width / 4, 3 * canvas.height / 4);
  context.stroke();
  context.closePath();

}

//animate - To animate particles by changing the X and Y points on each frame change. This results in smooth random movement. the dog animation function is also called.

function animate() {

  context.clearRect(0, 0, canvas.width, canvas.height);
  drawRect();

  
  Dog.animateDog();

  reqAnimFrame(function(t) {
    animate();
  });
}

animate();


//The setInterval runs the Dog animation on a frame rate of 10fps

setInterval(function() {
  ++Dog.frameIndex;
  count++;
  Dog.animateDog();
}, 1000 / frameRate);