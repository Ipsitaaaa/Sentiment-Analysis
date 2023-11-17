import React, { useState } from 'react';
import './App.css';

const SentimentAnalyzer = () => {
  const [keyword, setKeyword] = useState('');
  const [sentimentScore, setSentimentScore] = useState(null);

  const getSentimentScore = () => {
    fetch('http://127.0.0.1:8000/get_score', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ keyword: keyword }),
    })
      .then(response => response.json())
      .then(data => {
        setSentimentScore(data.data);
      })
      .catch(error => {
        console.error('Error:', error);
      });
  };

  return (
    <div className="container">
      <h1>Sentiment Analyzer</h1>
      <input
        type="text"
        value={keyword}
        onChange={e => setKeyword(e.target.value)}
        placeholder="Enter keyword"
      />
      <button onClick={getSentimentScore}>Get score</button>
      <p>{sentimentScore !== null && `The sentiment score for ${keyword} is ${sentimentScore}`}</p>
    </div>
  );
};

export default SentimentAnalyzer;
