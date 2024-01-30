const express = require("express");
const app = express();
const fs = require('fs/promises'); 
const bodyParser = require('body-parser');
app.use(express.json());

const filePath = './leaderboard.json';
app.use(bodyParser.json());


app.post('/updatescore', async (req, res) => {
    try {
      // Read the current leaderboard data from the JSON file
      const fileContent = await fs.readFile(filePath, 'utf8');
      const leaderboardData = JSON.parse(fileContent);
  
      // Extract player name and score from the request body
      const { playerName, newScore } = req.body;
  
      // Find the player in the leaderboard
      const playerIndex = leaderboardData.players.findIndex(
        (player) => player.name === playerName
      );
  
      if (playerIndex !== -1) {
        // If the player exists, update the score
        leaderboardData.players[playerIndex].score = newScore;
      } else {
        // If the player doesn't exist, add a new player
        leaderboardData.players.push({
          name: playerName,
          score: newScore,
        });
      }
  
      // Save the updated leaderboard data back to the JSON file
      await fs.writeFile(filePath, JSON.stringify(leaderboardData, null, 2), 'utf8');
  
      res.status(200).json({ message: 'Player score updated successfully' });
    } catch (error) {
      console.error('Error updating player score:', error);
      res.status(500).json({ error: 'Internal Server Error' });
    }
  });
 
  app.get('/leaderboard', async (req, res) => {
    try {
      // Read the file using fs/promises
      const data = await fs.readFile('leaderboard.json', 'utf8');
  
      // Parse the JSON data
      const jsonData = JSON.parse(data);
  
      // Send the JSON data as a response
      res.status(200).json(jsonData);
    } catch (error) {
      console.error('Error reading or parsing JSON:', error);
      res.status(500).send('Internal Server Error');
    }
  });

 
const port = process.env.PORT || 5500;
 
app.listen(port, () => console.log(`Server hört an Port: ${port}`));